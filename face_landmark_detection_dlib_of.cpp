 /*
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

// calculate the variance between the current points and last points
double cal_dist_diff(std::vector<cv::Point2f> curPoints, std::vector<cv::Point2f> lastPoints) {
    double variance = 0.0;
    double sum = 0.0;
    std::vector<double> diffs;
    if (curPoints.size() == lastPoints.size()) {
        for (int i = 0; i < curPoints.size(); i++) {
            double diff = std::sqrt(std::pow(curPoints[i].x - lastPoints[i].x, 2.0) + std::pow(curPoints[i].y - lastPoints[i].y, 2.0));
            sum += diff;
            diffs.push_back(diff);
        }
        double mean = sum / diffs.size();
        for (int i = 0; i < curPoints.size(); i++) {
            variance += std::pow(diffs[i] - mean, 2);
        }
        return variance / diffs.size();
    }
    return variance;
}

// Shows the results that optimized with Optical Flow method only
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

        // Initialize the points of last frame
        std::vector<cv::Point2f> last_object;
        for (int i = 0; i < 68; ++i) {
            last_object.push_back(cv::Point2f(0.0, 0.0));
        }

        double scaling = 0.5;
        int flag = -1;
        int count = 0;

        // Initialize Optical Flow
        cv::Mat prevgray, gray;
        std::vector<cv::Point2f> prevTrackPts;
        std::vector<cv::Point2f> nextTrackPts;
        for (int i = 0; i < 68; i++) {
            prevTrackPts.push_back(cv::Point2f(0, 0));
        }

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
            cv::Mat face_2 = temp.clone();
            cv::Mat face_3 = temp.clone();
            cv::Mat frame = temp.clone();

            if (flag == -1) {
                cvtColor(frame, prevgray, CV_BGR2GRAY);
                const full_object_detection& d = shapes[0];
                for (int i = 0; i < d.num_parts(); i++) {
                    prevTrackPts[i].x = d.part(i).x();
                    prevTrackPts[i].y = d.part(i).y();
                }
                flag = 1; 
            }

            if (shapes.size() == 1) {
                cvtColor(frame, gray, CV_BGR2GRAY);
                if (prevgray.data) {
                    std::vector<uchar> status;
                    std::vector<float> err;
                    calcOpticalFlowPyrLK(prevgray, gray, prevTrackPts, nextTrackPts, status, err);
                    std::cout << "variance:" << cal_dist_diff(prevTrackPts, nextTrackPts) << std::endl;
                    if (cal_dist_diff(prevTrackPts, nextTrackPts) > 1.0) {
                        const full_object_detection& d = shapes[0];
                        for (int i = 0; i < d.num_parts(); i++) {
                            cv::circle(frame, cv::Point2f(d.part(i).x(), d.part(i).y()), 2, cv::Scalar(0, 255, 255), -1);
                            nextTrackPts[i].x = d.part(i).x();
                            nextTrackPts[i].y = d.part(i).y();
                        }
                    } else {
                        for (int i = 0; i < nextTrackPts.size(); i++) {
                            cv::circle(frame, nextTrackPts[i], 2, cv::Scalar(0, 0, 255), -1);
                        }
                    }
                    std::swap(prevTrackPts, nextTrackPts);
                    //prevTrackPts = nextTrackPts;
                    std::swap(prevgray, gray);
                    }
                }
            
            cv::imshow("OpticalFlow", frame);

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
