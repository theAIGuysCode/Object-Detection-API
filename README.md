# Yolov3 Object Detection with Tensorflow 2.0 (APIs and Detections)
Yolov3 is an algorithm that uses deep convolutional neural networks to perform object detection. This repository implements Yolov3 using TensorFlow 2.0 and creates two easy-to-use APIs that you can integrate into web or mobile applications. <br>

## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-object-detection-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-object-detection-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU)
```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.

### Using Custom trained weights
<strong> Learn How To Train Custom YOLOV3 Weights Here: https://www.youtube.com/watch?v=zJDUhGL26iU </strong>

Add your custom weights file to weights folder and your custom .names file into data/labels folder.
  
### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!

```
# yolov3
python load_weights.py

# yolov3-tiny
python load_weights.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf --tiny
```

After executing one of the above lines, you should see .tf files in your weights folder.

## Running the model
Now you can run the model using `detect.py` script. 

Don't forget to set the IoU (Intersection over Union) and Confidence Thresholds within your yolov3-tf2/models.py file

### Usage examples
Let's run an example or two using sample images found within the data/images folder. 
```bash
# yolov3
python detect.py --images "data/images/dog.jpg, data/images/office.jpg"

# yolov3-tiny
python detect.py --weights ./weights/yolov3-tiny.tf --tiny --images "data/images/dog.jpg"

# webcam
python detect_video.py --video 0

# video file
python detect_video.py --video data/video/paris.mp4 --weights ./weights/yolov3-tiny.tf --tiny

# video file with output saved (can save webcam like this too)
python detect_video.py --video path_to_file.mp4 --output ./detections/output.avi
```
Then you can find the detections in the `detections` folder.
<br>
You should see these two images saved for running the first command.
```
detection1.jpg
```
![demo](https://github.com/theAIGuysCode/Object-Detection-API/blob/master/detections/detection1.jpg)
```
detection2.jpg
```
![demo](https://github.com/theAIGuysCode/Object-Detection-API/blob/master/detections/detection2.jpg)

### Video example
![demo](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detections.gif)

## Acknowledgments
* [Yolov3 TensorFlow 2 Amazing Implementation](https://github.com/zzh8829/yolov3-tf2)
* [Another Yolov3 TensorFlow 2](https://github.com/heartkilla/yolo-v3)
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
