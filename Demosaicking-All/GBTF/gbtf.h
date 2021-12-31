#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
void bayer_split(cv::Mat&, cv::Mat&);
void demosaic_gbtf(cv::Mat& Bayer, cv::Mat& Dst);