# GOP 

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