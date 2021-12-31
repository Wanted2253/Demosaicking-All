#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "RI/ri.h"
#include "GBTF/gbtf.h"
#include "MLRI/mlri.h"
#include "TIPRI/tipri.h"
//#include "ari.h"
using namespace std;
using namespace cv;

void Demosaic(cv::Mat& img, cv::Mat& Dst, int BayerPatternFlag,int methodflag);