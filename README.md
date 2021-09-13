# GOP 
We propose the GoP-mode acceleration technology to remove temporal redundancy. The current object detection neural network is a layer-based architecture from spatial domain point of view. From the temporal domain point of view, it is frame-based. Each frame must pass through all neural network layers for calculation. In addition to a large number of feature maps and weight values, there are many intermediate values of calculations that must be repeatedly sent in and out of the chip, resulting in slow computation speed. Nevertheless, there is a very high amount of redundancy existed between each frame of a video film. If it can be removed, a lot of calculation and data transmission can be saved without affecting the detection accuracy. 

Our idea is to adopt a method similar to Group-of-Picture (GoP) in video compression when detecting objects in video sequences. Only a small number of key frames are detected completely, i.e., intra frames (I-frames), in a group. The remainders of the group, i.e., inter frames (P-frames), are predicted based on these few key picture detection results. As shown in Fig. 3, each GoP contains one I-frame and several P-frames. Since the execution time of prediction is much shorter than that of detection, the value of frames per second (fps) has been greatly increased while remaining the overall detection accuracy the same.

<div align=center><img src="https://user-images.githubusercontent.com/50125053/133035858-c06c9e41-94fc-44bf-b610-1ec98d8b54f4.png"></div>

## Source
* [Train](https://github.com/ZQPei/deep_sort_pytorch)
* [Demo in Jetson Nano](https://github.com/jkjung-avt/tensorrt_demos)
## Download dataset
訓練資料: [MARS](http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html)

## Download tracker code
```bash=+
git clone https://github.com/ZQPei/deep_sort_pytorch.git
cd gop/torch2trt
cp *.py ~/deep_sort_pytorch/deep_sort/deep/
cd ~/
cd deep_sort_pytorch/deep_sort/deep/
```
## Train
```bash=+
python3 train.py --data-dir Path_of_MARS 
```
## Pytorch to TensorRT
pytorch to onnx
```bash=+
python3 torchtonnx.py 
```
onnx to TRT
```bash=+
python3 test2.py
```
## Download Detector code (TensorRT)

```bash=+
git clone https://github.com/jkjung-avt/tensorrt_demos.git
cp ~/gop/GOP_mode_tracker/.* ~/tensorrt_demos/
```
## detect & tracker
use web cam
```bash=+
python3 trt_yoloV2.py -m fp16/yoloV3-416(path_of_your_detection_model_in_./yolo/) --usb 0
```
use video
```bash=+
python3 trt_yoloV2.py -m fp16/yoloV3-416(path_of_your_detection_model_in_./yolo/) --video video_path
```
