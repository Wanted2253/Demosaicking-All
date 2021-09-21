#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
Mat guided_filter_mlri(const Mat& originG, const Mat& originR, const Mat& mask, const Mat& originI, const Mat& originP, const Mat& M, int h = 2, int v = 2, float eps = 0.0);