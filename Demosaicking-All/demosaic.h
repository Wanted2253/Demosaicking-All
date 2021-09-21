#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ri.h"
#include "gbtf.h"
#include "mlri.h"
#include "tipri.h"
//#include "ari.h"
using namespace std;
using namespace cv;

void Demosaic(cv::Mat& img, cv::Mat& Dst, int BayerPatternFlag,int methodflag);