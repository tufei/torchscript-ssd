# TorchScript on NVIDIA Jetson Nano

This is a demo application that shows how to run a network using TorchScripts
dumped from other platforms along with LibTorch C++ API.

The application has been tested using the NVIDIA Jetson Nano SDK. The performance
is actually not that good as it takes about 4 seconds to run MobileNetV2 based
SSD network with input resolution of 300x300.

This project is supposed to be used together with
[pytorch-ssd](https://github.com/tufei/pytorch-ssd), which can dump the TorchScript.

## Prerequisites

* The NVIDIA [Jetson Nano Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano-developer-kit/)
* NVIDIA [Jetpack SD card image](https://developer.nvidia.com/embedded/jetpack)
* Python wheels for Jetson Nano that provides [LibTorch](https://nvidia.box.com/v/torch-stable-cp36-jetson-jp42)
* GCC, Cmake, OpenCV, Boost, all of which can be installed from the Ubuntu 18.04
official repository

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=<your libtorch path> ..
```

## Run inference

```bash
./ts_ssd -s mb2-ssd-lite-mp-0_686.pt -l voc-model-labels.txt -p 0.5 -i messi.jpg
```

You should see message like the following:

```bash
–torch-script specified with value = mb2-ssd-lite-mp-0_686.pt
–labels specified with value = voc-model-labels.txt
–probability-threshold specified with value = 0.5
–input-file specified with value = ../libtorch-yolov3/imgs/messi.jpg
cuDNN: Yes
CUDA: Yes
Loaded TorchScript mb2-ssd-lite-mp-0_686.pt
Start inferencing ...
Original image size [width, height] = [1296, 729]
Inference done in 4305 ms
Tensor  shape: {3000 21}
Tensor  shape: {3000 4}
Class index [15]: person
Tensor  shape: {2 5}
```

## Example output

![Example of Mobile SSD](detected.jpg  "Example of Mobile SSD(Courtesy of https://github.com/walktree/libtorch-yolov3/blob/master/imgs/messi.jpg for the image.)")
