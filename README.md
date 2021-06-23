# Computer Pointer Controller

In this project, we will use a gaze detection model to control the mouse pointer of computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.

We are using four pre-trained models from the Intel Pre-trained Models Zoo:
* [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial LandMarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


## Project Set Up and Installation
**First step**
Make sure you have the OpenVINO toolkit installed on your system. This project is based on [Intel OpenVINO 2021.3.394](https://docs.openvinotoolkit.org/latest/index.html) toolkit, so if you don't have it, make sure to install it first before continue with the next steps.

**Second step**
You have to install the pretrained models needed for this project. The following instructions are for mac.
First you have to initialize the openVINO environment
```bash
source /opt/intel/openvino/bin/setupvars.sh
```
You have to run the above command every time you open an new terminal window.
We need the following models
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

To download them run the following commands after you have created a folder with name `model` and got into it.
**Face Detection Model**
```bash
$ python3 /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader/downloader.py --name face-detection-0200 --output_dir Computer-Pointer-Control/models
```
**Facial Landmarks Detection**
```bash
$ python3 /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir Computer-Pointer-Control/models
```
**Head Pose Estimation**
```bash
$ python3 /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir Computer-Pointer-Control/models
```
**Gaze Estimation Model**
```bash
$ python3 /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir Computer-Pointer-Control/models
```

**Third step**
Install the requirements:
```bash
$ pip3 install -r requirements.txt
```

**Project structure**
```bash
|
|--bin
    |--demo.mp4
|--model
    |--intel
        |--face-detection-adas-binary-0001
        |--gaze-estimation-adas-0002
        |--head-pose-estimation-adas-0001
        |--landmarks-regression-retail-0009
|--src
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--input_feeder.py
    |--main.py
    |--mouse_controller.py
    |--README.md
    |--requirements.txt
```


## Demo
Step 1:  Go back to the project directory src folder
 
        `cd src/` 
Step 2: Run below commands to execute the project
 * Run on CPU
 ```
python main.py -i ../bin/demo.mp4 -fd ../models/intel/face-detection-0200/FP16-INT8/face-detection-0200 -hp ../models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001 -fl ../models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009 -ge ../models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002 -d CPU -disp TRUE -pt 0.6

```
## Documentation
Below are the command line arguments needed and there brief use case.

Argument|Type|Description
| ------------- | ------------- | -------------
-fd | Required | Path to a face detection model.
-fl | Required | Path to a facial landmarks detection model.
-hp| Required | Path to a head pose estimation model.
-ge| Required | Path to a gaze estimation model.
-i| Required | Path to image or video file or WEBCAM.
-disp| Optional | Flag to display the outputs of the intermediate models.
-pt | Optional | Specify confidence threshold which the value here in range(0, 1), default=0.6
-d | Optional | Provide the target device: CPU / GPU / VPU / FPGA


## Benchmarks
Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
 The Performance tests were run on HP Laptop with **Intel® Core™ i3-8350K CPU @ 4.00GHz × 4** and **16 GB Ram**

#### CPU

| Properties  | FP32        | FP16        | INT8        |
| ------------| ----------- | ----------- | ----------- |
|Model Loading| 0.19139     | 0.183228    | 0.35699     |
|Infer Time   | 75.9599     | 75.77568    | 68.5793     |
|FPS          | 0.77672     | 0.77861fps  | 0.8603fps   |


## Results
We notice the models with low precisions generally tend to give better inference time, but it still difficult to give an exact measures as the time spent depend of the performance of the machine used in that given time when running the application. Also we notice that there is a difference between the same model with different precisions.

The models with low precisions are more lightweight than the models with high precisons, so this makes the execution of the network more fast.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
