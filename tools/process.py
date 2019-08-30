from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import tensorflow as tf
import numpy as np
import tfimage as im
import threading
import time
import cv2
from skimage.morphology import thin

edge_pool = None


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--operation", required=True, choices=["grayscale", "resize", "blank", "combine", "edges", "skeletonize"])
parser.add_argument("--workers", type=int, default=1, help="number of workers")
# resize
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
# combine
parser.add_argument("--b_dir", type=str, help="path to folder containing B images for combine operation")
# edges
parser.add_argument("--crop", action="store_true", help="crop the image before edge detection. Only works when background is white.")
parser.add_argument("--crop_dir", help="path for cropped original images")

a = parser.parse_args()

def resize(src):
    height, width, _ = src.shape
    dst = src
    if height != width:
        if a.pad:
            size = max(height, width)
            # pad to correct ratio
            oh = (size - height) // 2
            ow = (size - width) // 2
            dst = im.pad(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
        else:
            # crop to correct ratio
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            dst = im.crop(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

    assert(dst.shape[0] == dst.shape[1])

    size, _, _ = dst.shape
    if size > a.size:
        dst = im.downscale(images=dst, size=[a.size, a.size])
    elif size < a.size:
        dst = im.upscale(images=dst, size=[a.size, a.size])
    return dst


def blank(src):
    height, width, _ = src.shape
    if height != width:
        raise Exception("non-square image")

    image_size = width
    size = int(image_size * 0.3)
    offset = int(image_size / 2 - size / 2)

    dst = src
    dst[offset:offset + size,offset:offset + size,:] = np.ones([size, size, 3])
    return dst


def combine(src, src_path):
    if a.b_dir is None:
        raise Exception("missing b_dir")

    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(a.b_dir, basename + ext)
        if tf.io.gfile.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")
    
    # convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)


def grayscale(src):
    return im.grayscale_to_rgb(images=im.rgb_to_grayscale(images=src))


def crop_and_resize(src, return_gray = False):
    """
    crop edge image to discard white pad, and resize to training size
    based on: https://stackoverflow.com/questions/48395434/how-to-crop-or-remove-white-background-from-an-image
    [OBS!] only works on image with white background
    """
    height, width, _ = src.shape

    # (1) Convert to gray, and threshold
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # (4) Crop
    x, y, w, h = cv2.boundingRect(cnt)
    x_1 = max(x, x - 10)
    y_1 = max(y, y - 10)
    x_2 = min(x+w, width)
    y_2 = min(y+h, height)
    if return_gray:
        dst = gray[y_1:y_2, x_1:x_2]
    else:
        dst = src[y_1:y_2, x_1:x_2]
    # pad white to resize
    height = int(max(0, w - h) / 2.0)
    width = int(max(0, h - w) / 2.0)
    padded = cv2.copyMakeBorder(dst, height, height, width, width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return cv2.resize(padded, (a.size, a.size), interpolation=cv2.INTER_NEAREST)


def edges(src):
    src = np.asarray(src * 255, np.uint8)
    if a.crop:
        src = crop_and_resize(src)
    # detect edges based on Canny Edge Dection
    edge = cv2.bitwise_not(cv2.Canny(src, 80, 130))
    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    if a.crop:
        return np.asarray(src/255., np.float32), dst
    else:
        return dst


def skeletonize_edge(src):
    # Process sketch to fit input. Only used for test input
    src = np.asarray(src * 255, np.uint8)
    # Crop the sketch and minimize white padding.
    cropped = crop_and_resize(src, return_gray=True)
    # Skeletonize the lines
    skeleton = thin(cv2.bitwise_not(cropped))
    final = np.asarray(1 - np.float32(skeleton))
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

def process(src_path, dst_path):
    src = im.load(src_path)
    if a.operation == "edges":
        if a.crop:
            name = dst_path.split("/")[-1]
            src, dst = edges(src)
            im.save(src, os.path.join(a.crop_dir, name))
        else:
            dst = edges(src)
    elif a.operation == "grayscale":
        dst = grayscale(src)
    elif a.operation == "resize":
        dst = resize(src)
    elif a.operation == "blank":
        dst = blank(src)
    elif a.operation == "combine":
        dst = combine(src, src_path)
    elif a.operation == "skeletonize":
        dst = skeletonize_edge(src)
    else:
        raise Exception("invalid operation")

    im.save(dst, dst_path)


complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def main():
    if not tf.io.gfile.exists(a.output_dir):
        tf.io.gfile.makedirs(a.output_dir)
    if a.operation == "edges" and a.crop:
        try:
            if not tf.io.gfile.exists(a.crop_dir):
                tf.io.gfile.makedirs(a.crop_dir)
        except Exception as e:
            raise Exception("invalid crop_dir: {:s}".format(e))

    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in im.find(a.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".png")
        if tf.io.gfile.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)
    
    print("skipping %d files that already exist" % skipped)
            
    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()



    if a.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path)
                complete()
    else:
        queue = tf.train.input_producer(zip(src_paths, dst_paths), shuffle=False, num_epochs=1)
        dequeue_op = queue.dequeue()

        def worker(coord):
            with sess.as_default():
                while not coord.should_stop():
                    try:
                        src_path, dst_path = sess.run(dequeue_op)
                    except tf.errors.OutOfRangeError:
                        coord.request_stop()
                        break

                    process(src_path, dst_path)
                    complete()

        # init epoch counter for the queue
        local_init_op = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(local_init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(a.workers):
                t = threading.Thread(target=worker, args=(coord,))
                t.start()
                threads.append(t)
            
            try:
                coord.join(threads)
            except KeyboardInterrupt:
                coord.request_stop()
                coord.join(threads)

main()
