## Introduction
This repo generally shows the example usages about utilizing Kalman Filter on facial landmark detection based on [dlib](http://dlib.net/). It also reveals an idea about optimizing the correctness of facial landmark detection from dlib based on a combined method of Kalman Filter, Optical Flow and dlib.

### Demo Link
The following link shows the results from [face_landmark_detection_dlib_of_kf_mixed.cpp](https://github.com/zhucebuliaolongchuan/KalmanFilter/blob/master/face_landmark_detection_dlib_of_kf_mixed.cpp)

https://www.youtube.com/watch?v=x1vGY6HgRHw

### Dependencies
* OpenCV (>=2.4.3)
* dlib

### Usage
#### Compile
```
$ mkdir build
$ cd build
$ cmake .. -DUSE_AVX_INSTRUCTIONS=1
$ make -j8
```

#### Run
```
$ ./face_landmark_detection
```
