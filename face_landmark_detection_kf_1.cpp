/**
landmarks in images read from a webcam and points that drew by the program.
 */

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <stdio.h>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace dlib;

void print_usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "./face_landmark_detection [path/to/shape_predictor_68_face_landmarks.dat]" << std::endl;
}

// example usage for the single-point kalman filter
int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            print_usage();
            return 0;
        }

        std::cout << "Press q to exit." << std::endl;

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Unable to connect to camera" << std::endl;
            return 1;
        }

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;

        if (argc == 2) {
            deserialize(argv[1]) >> pose_model;
        } else {
            deserialize("../../data/shape_predictor_68_face_landmarks.dat") >> pose_model;
        }

        std::vector<cv::Point2f> last_object;
        for (int i = 0; i < 68; ++i) {
            last_object.push_back(cv::Point2i(0.0, 0.0));
        }

        double scaling = 0.5;
        int flag = -1;
        cv::Point2f curPoint(0.0, 0.0);

        // Kalman Filter Setup (One Pont Test)
        const int stateNum = 4;
        const int measureNum = 2;

        KalmanFilter KF(stateNum, measureNum, 0);
        Mat state(stateNum, 1, CV_32FC1);
        Mat processNoise(stateNum, 1, CV_32F);
        Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

        // Generate a matrix randomly
        randn(state, Scalar::all(0), Scalar::all(0.1));
        KF.transitionMatrix = *(Mat_<float>(4, 4) <<   
                                1,0,1,0,
                                0,1,0,1,   
                                0,0,1,0,   
                                0,0,0,1 );

        //!< measurement matrix (H) Measurement Model  
        setIdentity(KF.measurementMatrix);
  
        //!< process noise covariance matrix (Q)  
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
          
        //!< measurement noise covariance matrix (R)  
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));

        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix  
        setIdentity(KF.errorCovPost, Scalar::all(1));
    
        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

        // Grab and process frames until the main window is closed by the user.
        while(true) {
            // Grab a frame
            cv::Mat raw;
            cap >> raw;
            
            // Resize
            cv::Mat tmp;
            cv::resize(raw, tmp, cv::Size(), scaling, scaling);

            //Flip
            cv::Mat temp;
            cv::flip(tmp, temp, 1);

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces, load the vertexes as vector 
            std::vector<dlib::rectangle> faces = detector(cimg);

            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                shapes.push_back(pose_model(cimg, faces[i]));
            }

            // We cannot modify temp so we clone a new one
            cv::Mat face = temp.clone();
            // We strict to detecting one face
            cv::Mat face_2 = temp.clone();
            
            // Simple Filter
            if (shapes.size() == 1) {
                const full_object_detection& d = shapes[0];
                if (flag == -1) {
                    for (int i = 0; i < d.num_parts(); i++) {
                        cv::circle(face, cv::Point(d.part(i).x(), d.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
                        std::cout << i << ": " << d.part(i) << std::endl;
                    }
                    flag = 1;
                } else {
                     for (int i = 0; i < d.num_parts(); i++) {
                        cv::circle(face, cv::Point2f(d.part(i).x() * 0.5 + last_object[i].x * 0.5, d.part(i).y() * 0.5 + last_object[i].y * 0.5), 2, cv::Scalar(0, 0, 255), -1);
                        std::cout << i << ": " << d.part(i) << std::endl;
                    }
                }
                for (int i = 0; i < d.num_parts(); i++) {
                    last_object[i].x = d.part(i).x();
                    last_object[i].y = d.part(i).y();
                }
            }

            // No Filter
            if (shapes.size() == 1) {
                const full_object_detection& d = shapes[0];
                curPoint.x = d.part(20).x();
                curPoint.y = d.part(20).y();
                for (int i = 0; i < d.num_parts(); i++) {
                    cv::circle(face_2, cv::Point2f(int(d.part(i).x()), int(d.part(i).y())), 2, cv::Scalar(0, 255, 255), -1);
                    std::cout << i << ": " << d.part(i) << std::endl;
                }
            }

            // One Point Kalman Filter
            cv::Point2f statePt = cv::Point2f(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
            // Kalman Prediction
            Mat prediction = KF.predict();
            cv::Point2f predict_point = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));

            // Update Measurement
            measurement.at<float>(0) = (float)curPoint.x;
            measurement.at<float>(1) = (float)curPoint.y;

            measurement += KF.measurementMatrix * state;

            // Correct Measurement
            KF.correct(measurement);

            // Show the one-point kalman filter
            cv::circle(face, predict_point, 2, cv::Scalar(255, 0, 0), -1);
            cv::circle(face_2, predict_point, 2, cv::Scalar(255, 0, 0), -1);
            //std::cout << predict_point << std::endl;


            // Display the frame with landmarks
            cv::imshow("face", face);
            cv::imshow("face_2", face_2);

            char key = cv::waitKey(1);
            if (key == 'q') {
                break;
            }
        }
    } catch(serialization_error& e) {
        print_usage();
    } catch(std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}