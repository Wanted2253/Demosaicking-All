#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "guidedfilter.h"
using namespace std;
using namespace cv;
void bayer_splitri(cv::Mat&, cv::Mat&);
void bayer_maskri(cv::Mat&, cv::Mat&);
void demosaic_residual(cv::Mat&, cv::Mat&, float sigma = 1.0);
