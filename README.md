# pix2pix-tensorflow

The repo is a Tensorflow implementation of pix2pix, forked from [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). The readme file also keeps some descriptions from the original repo.
Original article about the implementation please check: [article](https://affinelayer.com/pix2pix/)

The model is used for transfer-learning, targeting at generating chairs from sketch.
Additional functions are implemented based on the model:
- Training data creation: Replace HED edge detector to Canny edge detector, implemented with OpenCV.
- Data augmentation: add random rotation of images for training set.  
- Export data: Add export method for model deployment on Google AI-platform.

## Setup

### Prerequisites
`pip install -r requirement.txt`

## Data preparation
### Format
The data format used by this program is the same as the original pix2pix format, which consists of images of input and desired output side by side like:

<img src="docs/ab.png" width="256px"/>

### Datasets
Dataset used for training are based on pairs of images with a detected edge as sketch and a target image as a chair with white background.
Datasets used for learning includes:
- [Interior Design Dataset - IKEA](https://github.com/IvonaTau/ikea)
- [sketch-photo datasets](https://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html) by Yu, Qian et al.
- [Ikea website](https://www.ikea.com/se/sv/)
- [Seeing 3D chairs](https://www.di.ens.fr/willow/research/seeing3Dchairs/) by Mathieu Aubry et al.

### Training pair creation
Original Image clean up and edge detection:
```sh
python tools/process.py \
  --input_dir data/<original images> \
  --operation edges \
  --output_dir data/<detected edge path> \
  --crop \
  --crop_dir data/<cropped images>
```

Combine detected edges with cropped original images:
```sh
python tools/process.py \
  --input_dir data/<detected edge path> \
  --b_dir data/<cropped images> \
  --operation combine \
  --output_dir data/<combined images>
```

## Training

### Image Pairs

For normal training with image pairs, you need to specify which directory contains the training images, and which direction to train on.  The direction options are `AtoB` or `BtoA`
```sh
python pix2pix.py \
  --mode train \
  --output_dir models/chair-gan \
  --max_epochs 200 \
  --save_freq 1000 \
  --batch_size 10 \ 
  --input_dir data/<combined images> \
  --which_direction AtoB
  --checkpoint pre-trained/<pretrained model/ checkpoint to continue> \
```
In the demo of chair generation,  the model performs a transfer-learning based on [shoes generation](https://mega.nz/#!u9pnmC4Q!2uHCZvHsCkHBJhHZ7xo5wI-mfekTwOK8hFPy0uBOrb4)

## Testing

Testing is done with `--mode test`.  You should specify the checkpoint to use with `--checkpoint`, this should point to the `output_dir` that you created previously with `--mode train`:

```sh
python pix2pix.py \
  --mode test \
  --output_dir test/chair-gan \
  --input_dir data/<test images (same format as training)> \
  --checkpoint models/chair-gan
```

The testing mode will load some of the configuration options from the checkpoint provided so you do not need to specify `which_direction` for instance.

The test run will output an HTML file at `test/chair-gan/index.html` that shows input/output/target image sets

## Model export and Serving
Currently pre-processing of sketch data is not implemented as custom script which can be called by AI-platform. 
As a result, preprocessing is needed to be done separately, with steps:
1. Crop sketch image
1. Resizeing to 256 x 256
1. Skeletonization of sketch lines

Batch processing of the data can be done by:
``` sh
python tools/process.py \
  --input_dir data/<sketches> \
  --operation sketch\
  --output_dir data/<processed sketches>
```

For information of model export and serving, please check [notebook](notebooks/Export_model_and_serve.ipynb) 

## Unimplemented Features

The following models have not been implemented:
- Evaluation metric during training
- Test image pre-processing script for model deployment on AI-platform
- Update tensorflow and add support for AI-platform training

## Citation
If you use this code for your research, please cite the paper this code is based on: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation Using Conditional Adversarial Networks</a>:

```
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}
```
