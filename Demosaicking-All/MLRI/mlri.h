#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "guidedfiltermlri.h"
using namespace std;
using namespace cv;
void bayer_splitmlri(cv::Mat&, cv::Mat&);
void bayer_maskmlri(cv::Mat&, cv::Mat&);
void demosaic_mlri(cv::Mat&, cv::Mat&, float sigma = 1.0);