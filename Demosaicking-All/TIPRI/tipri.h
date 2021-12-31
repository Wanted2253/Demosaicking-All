#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "guidedfiltertipri.h"
using namespace std;
using namespace cv;
void bayer_splittipri(cv::Mat&, cv::Mat&);
void bayer_masktipri(cv::Mat&, cv::Mat&);
void demosaic_tipri(cv::Mat&, cv::Mat&, float sigma = 1.0);