import cv2
from skimage.morphology import thin
import numpy as np
import base64

def crop_and_resize(src):
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
    x_2 = min(x + w, width)
    y_2 = min(y + h, height)
    dst = gray[y_1:y_2, x_1:x_2]
    # pad white to resize
    height = int(max(0, w - h) / 2.0)
    width = int(max(0, h - w) / 2.0)
    padded = cv2.copyMakeBorder(dst, height, height, width, width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return cv2.resize(padded, (256, 256), interpolation=cv2.INTER_NEAREST)

def preprocess(data):
    img_b64 = data["img"]
    img_bytes=base64.b64decode(img_b64)
    src = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
    
    # Crop the sketch and minimize white padding.
    cropped = crop_and_resize(src)
    # Skeletonize the lines
    skeleton = thin(cv2.bitwise_not(cropped))
    final = np.asarray(1 - np.float32(skeleton))
    fixed_channel = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    _, img_png = cv2.imencode('.png', fixed_channel)
    encoded_input_string = base64.b64encode(img_png.tobytes())
    return {'image_bytes': {"b64": encoded_input_string.decode()}} 