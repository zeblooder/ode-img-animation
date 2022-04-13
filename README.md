# ode-img-animation

This repository contains the source code for the thesis A new framework for Image animation with motion from a driving video.

## Installation

1. Clone this repository
2. Build Gaussian Sampling CUDA package

```shell
cd ./ode-img-animation/modules/resample2d_package
python setup.py install --user
```

3. Install the dependencies

```shell
pip install -r requirements.txt
```

4. Install torchdiffeq

```
pip install torchdiffeq
```

### YAML configs

There are several configuration (`config/dataset_name.yaml`) files one for each `dataset`.

### Training

We can only train the model by one device.

```shel
python run.py --config config/dataset_name.yaml
```

### Evaluation

before evaluating the model, add path of the model to corresponding configuration file.

```shell
python run.py --mode evaluation  --metrics "l1,LPIPS,PSNR,MS-/SSIM" --config config/dataset_name.yaml
```

### Animation Demo

```shell
python demo.py  --config config/dataset_name.yaml --driving_video path/to/driving --source_image path/to/source --checkpoint path/to/checkpoint --relative --adapt_scale
```

The result will be stored in `result.mp4`.

### Datasets

Please follow the [instructions](https://github.com/AliaksandrSiarohin/first-order-model) on how to download the dataset.
