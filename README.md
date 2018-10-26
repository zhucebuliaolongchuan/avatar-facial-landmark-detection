# Introduction
This repo generally shows the example usages about utilizing Kalman Filter on facial landmark detection based on [dlib](http://dlib.net/). It also reveals an idea about optimized the correctness of facial landmark detection based on a combined method from Kalman Filter, Optical Flow and Dlib.

## Demo Link
https://www.youtube.com/watch?v=x1vGY6HgRHw

## Dependencies
* OpenCV (>=2.4.3)
* dlib

## Usage
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