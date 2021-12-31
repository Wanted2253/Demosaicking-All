#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "RI/guidedfilter.h"
using namespace std;
using namespace cv;
void toSingleChannelari(cv::Mat& src, cv::Mat& dst) {
	if (src.channels() != 3) {
		std::cerr << "to_SingleChannel need 3 channel image" << std::endl;
		return;
	}
	transform(src, dst, cv::Matx13f(1, 1, 1));
	return;
}
//   0 1 
// 0 R G
// 1 G B
void bayer_splitari(cv::Mat& Bayer, cv::Mat& Dst) {
	if (Bayer.channels() != 1) {
		std::cerr << "bayer_split allow only 1 channel raw bayer image " << std::endl;
		return;
	}
	Dst = cv::Mat::zeros(Bayer.rows, Bayer.cols, CV_8UC3);
	int channelNum;

	for (int row = 0; row < Bayer.rows; row++) {
		for (int col = 0; col < Bayer.cols; col++) {
			if (row % 2 == 0) { // opencv: BGR
				channelNum = (col % 2 == 0) ? 2 : 1;//even rows -- even cols:R=channel:2; odd cols:G=channel:1 
			}
			else {
				channelNum = (col % 2 == 0) ? 1 : 0;// odd rowd -- even cols:G=channel:1; odd cols:B=channel:3
			}
			Dst.at<Vec3b>(row, col).val[channelNum] = Bayer.at<uchar>(row, col);
		}
	}
	return;
}
void bayer_maskari(cv::Mat& Bayer, cv::Mat& Dst) {
	Mat temp = cv::Mat::ones(Bayer.size(), CV_8U);
	bayer_splitari(temp, Dst);// or split(Bayer, temp)
}
void demosaic_ari(cv::Mat& Bayer, cv::Mat& Dst, float sigma = 1.0) {
	cv::Mat Src = Bayer.clone();
	if (Bayer.channels() == 1) { //input 1 channel -> 3 channel Bayer
		bayer_splitari(Bayer, Src); // or split(Bayer, Src);
	}

	Mat Src1ch;
	toSingleChannelari(Src, Src1ch); // or merge(Src, Src1ch);
	Src1ch.convertTo(Src1ch, CV_32F, 1.0 / 255.0); //normalize

	Mat tempMask;
	bayer_maskari(Src, tempMask);
	tempMask.convertTo(tempMask, CV_32F); //normalize
	vector<Mat> mask(3);
	split(tempMask, mask);
	tempMask.release();

	// split channel to BGR 
	Src.convertTo(Src, CV_32F, 1.0 / 255.0); //normalize
	vector<Mat> bgr(3);
	vector<Mat> finalBGR(3);
	split(Src, bgr);
	split(Src, finalBGR);

	// ==== 1.Green interpolation ===
	Mat maskGr = Mat::zeros(Src.rows, Src.cols, CV_32F);
	Mat maskGb = Mat::zeros(Src.rows, Src.cols, CV_32F);
	for (int row = 0; row < Src.rows; row++) {
		int col = 0;
		float* targetGr = maskGr.ptr<float>(row);
		float* targetGb = maskGb.ptr<float>(row);
		if (row % 2 == 0) {
			col = 1;
		}
		for (; col < Src.cols; col += 2) {
			if (row % 2 == 0) { // R G
				targetGr[col] = 1.0;
			}
			else { //G B
				targetGb[col] = 1.0;
			}
		}
	}

	float VHkernel[3] = { 0.5, 0, 0.5 }; //bilinear interpolation at 1D
	cv::Mat HK(1, 3, CV_32F, VHkernel);
	cv::Mat VK(3, 1, CV_32F, VHkernel);
	Mat rawH, rawV;
	filter2D(Src1ch, rawH, -1, HK); //original matlab uses'replicate' for all filter
	filter2D(Src1ch, rawV, -1, VK);
	// Mrh: mask of horizontal Rand G line
	// Mbh : mask of horizontal Band G line
	// Mrv : mask of vertical Rand G line
	// Mbv : mask of vertical Band G line
	Mat Mrh = mask[2] + maskGr;
	Mat Mbh = mask[0] + maskGb;
	Mat Mrv = mask[2] + maskGb;
	Mat Mbv = mask[0] + maskGr;
	// Creating guide images
	Mat GuideGR_H = bgr[1].mul(maskGr) + rawH.mul(mask[2]);
	Mat GuideGB_H = bgr[1].mul(maskGb) + rawH.mul(mask[0]);
	Mat GuideR_H = bgr[2] + rawH.mul(maskGr);
	Mat GuideB_H = bgr[0] + rawH.mul(maskGb);
	Mat GuideGR_V = bgr[1].mul(maskGr) + rawV.mul(mask[2]);
	Mat GuideGB_V = bgr[1].mul(maskGb) + rawV.mul(mask[0]);
	Mat GuideR_V = bgr[2] + rawV.mul(maskGb);
	Mat GuideB_V = bgr[0] + rawV.mul(maskGr);

	// Tentative image
	int h = 2; //horizontal
	int v = 1; //vertical
	float eps = 0;
	int h2 = 4;
	int v2 = 0;
	int itnum = 11;
	//Initialization of Horizontal and Vertical Criteria
	Mat RI_w2H = cv::Mat::ones(maskGr.size(), CV_8U) * pow(10,32);
	Mat RI_w2V = cv::Mat::ones(maskGr.size(), CV_8U) * pow(10, 32);
	Mat MLRI_w2H = cv::Mat::ones(maskGr.size(), CV_8U) * pow(10, 32);
	Mat MLRI_w2V = cv::Mat::ones(maskGr.size(), CV_8U) * pow(10, 32);
	// initial guide image for RI
	Mat RI_Guidegrh = GuideGR_H;
	Mat RI_Guidegbh = GuideGB_H;
	Mat RI_Guiderh = GuideR_H;
	Mat RI_Guidebh = GuideB_H;
	Mat RI_Guidegrv = GuideGR_V;
	Mat RI_Guidegbv = GuideGB_V;
	Mat RI_Guiderv = GuideR_V;
	Mat RI_Guidebv = GuideB_V;
	// initial guide image for MLRI
	Mat MLRI_Guidegrh = GuideGR_H;
	Mat MLRI_Guidegbh = GuideGB_H;
	Mat MLRI_Guiderh = GuideR_H;
	Mat MLRI_Guidebh = GuideB_H;
	Mat MLRI_Guidegrv = GuideGR_V;
	Mat MLRI_Guidegbv = GuideGB_V;
	Mat MLRI_Guiderv = GuideR_V;
	Mat MLRI_Guidebv = GuideB_V;

	// initialization of interpolated G values
	Mat RI_Gh = GuideGR_H + GuideGB_H;
	Mat RI_Gv = GuideGR_V + GuideGB_V;
	Mat MLRI_Gh = GuideGR_H + GuideGB_H;
	Mat MLRI_Gv = GuideGR_V + GuideGB_V;
	for (int i = 1; i <= itnum; i++);
	//Generate Horizontal and Vertical Estimates for RI
	Mat RI_tentativeGrh = guided_filter_modified(RI_Guiderh, RI_Guidegrh, Mrh, h, v, eps);
	Mat RI_tentativeGbh = guided_filter_modified(RI_Guidebh, RI_Guidegbh, Mbh, h, v, eps);
	Mat RI_tentativeRh = guided_filter_modified(RI_Guidegrh, RI_Guiderh, Mrh, h, v, eps);
	Mat RI_tentativeBh = guided_filter_modified(RI_Guidegbh, RI_Guidebh, Mbh, h, v, eps);
	Mat RI_tentativeGrv = guided_filter_modified(RI_Guiderv, RI_Guidegrv, Mrv, v, h, eps);
	Mat RI_tentativeGbv = guided_filter_modified(RI_Guidebv, RI_Guidegbv, Mbv, v, h, eps);
	Mat RI_tentativeRv = guided_filter_modified(RI_Guidegrv, RI_Guiderv, Mrv, v, h, eps);
	Mat RI_tentativeBv = guided_filter_modified(RI_Guidegbv, RI_Guidebv, Mbv, v, h, eps);
	//Horizontal Tentative Estimate by MLRI
	float FHkernel[5] = { -1 , 0 , 2 , 0 , -1 };
	Mat FHK(1, 5, CV_32F, FHkernel);
	Mat difRh,difGrh,difBh,difGbh;
	filter2D(MLRI_Guiderh, difRh, -1, FHK);
	filter2D(MLRI_Guidegrh, difGrh, -1, FHK);
	filter2D(MLRI_Guidebh, difBh, -1, FHK);
	filter2D(MLRI_Guidegbh, difGbh, -1, FHK);
	Mat MLRI_tentativeRh = guided_filter_mlri(MLRI_Guidegrh, MLRI_Guiderh, Mrh, difGrh, difRh, mask[2], h2, v2, eps);
	Mat MLRI_tentativeBh = guided_filter_mlri(MLRI_Guidegbh, MLRI_Guidebh, Mbh, difGbh, difBh, mask[0], h2, v2, eps);
	Mat MLRI_tentativeGrh = guided_filter_mlri(MLRI_Guiderh, MLRI_Guidegrh, Mrh, difRh, difGrh, maskGr, h2, v2, eps);
	Mat MLRI_tentativeGbh = guided_filter_mlri(MLRI_Guiderh, MLRI_Guidegrh, Mrh, difRh, difGrh, maskGb, h2, v2, eps);
	// vertical part
	float FVkernel[5] = { -1 , 0 , 2 , 0 , -1 };
	Mat FVK(5, 1, CV_32F, FVkernel);
	Mat difRv, difGrv, difBv, difGbv;
	filter2D(MLRI_Guiderv, difRh, -1, FVK);
	filter2D(MLRI_Guidegrv, difGrh, -1, FVK);
	filter2D(MLRI_Guidebv, difBh, -1, FVK);
	filter2D(MLRI_Guidegbv, difGbh, -1, FVK);
	Mat MLRI_tentativeRv = guided_filter_mlri(MLRI_Guidegrv, MLRI_Guiderv, Mrh, difGrv, difRv, mask[2], h2, v2, eps);
	Mat MLRI_tentativeBv = guided_filter_mlri(MLRI_Guidegbv, MLRI_Guidebv, Mbh, difGbv, difBv, mask[0], h2, v2, eps);
	Mat MLRI_tentativeGrv = guided_filter_mlri(MLRI_Guiderv, MLRI_Guidegrv, Mrh, difRv, difGrv, maskGr, h2, v2, eps);
	Mat MLRI_tentativeGbv= guided_filter_mlri(MLRI_Guiderv, MLRI_Guidegrv, Mrh, difRv, difGrv, maskGb, h2, v2, eps);

	//release Guided Image
	

	// Residual
	Mat RI_residualGrh = (bgr[1] - RI_tentativeGrh).mul(maskGr);
	Mat	RI_residualGbh = (bgr[1] - RI_tentativeGbh).mul(maskGb);
	Mat	RI_residualRh = (bgr[2] - RI_tentativeRh).mul(mask[2]);
	Mat	RI_residualBh = (bgr[0] - RI_tentativeBh).mul(mask[0]);
	Mat	RI_residualGrv = (bgr[1] - RI_tentativeGrv).mul(maskGb);//FLAG
	Mat	RI_residualGbv = (bgr[1] - RI_tentativeGbv).mul(maskGr);
	Mat	RI_residualRv = (bgr[2] - RI_tentativeRv).mul(mask[2]);
	Mat	RI_residualBv = (bgr[0] - RI_tentativeBv).mul(mask[0]);
	Mat	MLRI_residualGrh = (bgr[1] - MLRI_tentativeGrh).mul(maskGr);
	Mat	MLRI_residualGbh = (bgr[1] - MLRI_tentativeGbh).mul(maskGb);
	Mat	MLRI_residualRh = (bgr[2] - MLRI_tentativeRh).mul(mask[2]);
	Mat	MLRI_residualBh = (bgr[0] - MLRI_tentativeBh).mul(mask[0]);
	Mat	MLRI_residualGrv = (bgr[1] - MLRI_tentativeGrv).mul(maskGb);
	Mat	MLRI_residualGbv = (bgr[1] - MLRI_tentativeGbv).mul(maskGr);
	Mat	MLRI_residualRv = (bgr[2] - MLRI_tentativeRv).mul(mask[2]);
	Mat	MLRI_residualBv = (bgr[0] - MLRI_tentativeBv).mul(mask[0]);
	// Residual interpolation
	float VH2kernel[3] = { 0.5, 1, 0.5 }; //Horizontal and Vertical Linear Interpolation
	cv::Mat HK2(1, 3, CV_32F, VHkernel);
	cv::Mat VK2(3, 1, CV_32F, VHkernel);
	filter2D(RI_residualGrh, RI_residualGrh, -1, HK2);
	filter2D(RI_residualGbh, RI_residualGbh, -1, HK2);
	filter2D(RI_residualRh, RI_residualRh, -1, HK2);
	filter2D(RI_residualBh, RI_residualBh, -1, HK2);
	filter2D(RI_residualGrv, RI_residualGrv, -1, VK2);
	filter2D(RI_residualGbv, RI_residualGbv, -1, VK2);
	filter2D(RI_residualRv, RI_residualRv, -1, VK2);
	filter2D(RI_residualBv, RI_residualBv, -1, VK2);
	filter2D(MLRI_residualGrh, MLRI_residualGrh, -1, HK2);
	filter2D(MLRI_residualGbh, MLRI_residualGbh, -1, HK2);
	filter2D(MLRI_residualRh, MLRI_residualRh, -1, HK2);
	filter2D(MLRI_residualBh, MLRI_residualBh, -1, HK2);
	filter2D(MLRI_residualGrv, MLRI_residualGrv, -1, VK2);
	filter2D(MLRI_residualGbv, MLRI_residualGbv, -1, VK2);
	filter2D(MLRI_residualRv, MLRI_residualRv, -1, VK2);
	filter2D(MLRI_residualBv, MLRI_residualBv, -1, VK2);
	// Adding tentative to residual
	Mat RI_Grh = (RI_tentativeGrh + RI_residualGrh).mul(mask[2]);
	Mat RI_Gbh = (RI_tentativeGbh + RI_residualGbh).mul(mask[0]);
	Mat RI_Rh = (RI_tentativeRh + RI_residualRh).mul(maskGr);
	Mat RI_Bh = (RI_tentativeBh + RI_residualBh).mul(maskGb);
	Mat RI_Grv = (RI_tentativeGrv + RI_residualGrv).mul(mask[2]);
	Mat RI_Gbv = (RI_tentativeGbv + RI_residualGbv).mul(mask[0]);
	Mat RI_Rv = (RI_tentativeRv + RI_residualRv).mul(maskGb);
	Mat RI_Bv = (RI_tentativeBv + RI_residualBv).mul(maskGr);
	Mat MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh).mul(mask[2]);
	Mat MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh).mul(mask[0]);
	Mat MLRI_Rh = (MLRI_tentativeRh + MLRI_residualRh).mul(maskGr);
	Mat MLRI_Bh = (MLRI_tentativeBh + MLRI_residualBh).mul(maskGb);
	Mat MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv).mul(mask[2]);
	Mat MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv).mul(mask[0]);
	Mat MLRI_Rv = (MLRI_tentativeRv + MLRI_residualRv).mul(maskGr);
	Mat MLRI_Bv = (MLRI_tentativeBv + MLRI_residualBv).mul(maskGr);
	
	// calculate iteration criteria(Eq.(3))
	Mat RI_criGrh = (RI_Guidegrh - RI_tentativeGrh).mul(Mrh);
	Mat RI_criGbh = (RI_Guidegbh - RI_tentativeGbh).mul(Mbh);
	Mat RI_criRh = (RI_Guiderh - RI_tentativeRh).mul(Mrh);
	Mat RI_criBh = (RI_Guidebh - RI_tentativeBh).mul(Mbh);
	Mat RI_criGrv = (RI_Guidegrv - RI_tentativeGrv).mul(Mrv);
	Mat RI_criGbv = (RI_Guidegbv - RI_tentativeGbv).mul(Mbv);
	Mat RI_criRv = (RI_Guiderv - RI_tentativeRv).mul(Mrv);
	Mat RI_criBv = (RI_Guidebv - RI_tentativeBv).mul(Mbv);
	Mat MLRI_criGrh = (MLRI_Guidegrh - MLRI_tentativeGrh).mul(Mrh);
	Mat MLRI_criGbh = (MLRI_Guidegbh - MLRI_tentativeGbh).mul(Mbh);
	Mat MLRI_criRh = (MLRI_Guiderh - MLRI_tentativeRh).mul(Mrh);
	Mat MLRI_criBh = (MLRI_Guidebh - MLRI_tentativeBh).mul(Mbh);
	Mat MLRI_criGrv = (MLRI_Guidegrv - MLRI_tentativeGrv).mul(Mrv);
	Mat MLRI_criGbv = (MLRI_Guidegbv - MLRI_tentativeGbv).mul(Mbv);
	Mat MLRI_criRv = (MLRI_Guiderv - MLRI_tentativeRv).mul(Mrv);
	Mat MLRI_criBv = (MLRI_Guidebv - MLRI_tentativeBv).mul(Mbv);

	// calculate gradient of iteration criteria
	float VH3kernel[3] = { -1, 0, 1 }; //bilinear interpolation at 1D
	cv::Mat HK3(1, 3, CV_32F, VHkernel);
	cv::Mat VK3(3, 1, CV_32F, VHkernel);
	Mat RI_difcriGrh, RI_difcriGbh, RI_difcriRh, RI_difcriBh, MLRI_difcriGrh, MLRI_difcriGbh, MLRI_difcriRh, MLRI_difcriBh;
	Mat RI_difcriGrv, RI_difcriGbv, RI_difcriRv, RI_difcriBv, MLRI_difcriGrv, MLRI_difcriGbv, MLRI_difcriRv, MLRI_difcriBv;
	filter2D(RI_criGrh, RI_difcriGrh, -1, HK3);
	filter2D(RI_criGbh, RI_difcriGbh, -1, HK3);
	filter2D(RI_criRh, RI_difcriRh, -1, HK3);
	filter2D(RI_criBh, RI_difcriBh, -1, HK3);
	filter2D(RI_criGrv, RI_difcriGrv, -1, VK3);
	filter2D(RI_criGbv, RI_difcriGbv, -1, VK3);
	filter2D(RI_criRv, RI_difcriGrh, -1, VK3);
	filter2D(RI_criBv, RI_difcriGrh, -1, VK3);
	filter2D(MLRI_criGrh, MLRI_difcriGrh, -1, HK3);
	filter2D(MLRI_criGbh, MLRI_difcriGbh, -1, HK3);
	filter2D(MLRI_criRh, MLRI_difcriRh, -1, HK3);
	filter2D(MLRI_criBh, MLRI_difcriBh, -1, HK3);
	filter2D(MLRI_criGrv, MLRI_difcriGrv, -1, VK3);
	filter2D(MLRI_criGbv, MLRI_difcriGbv, -1, VK3);
	filter2D(MLRI_criRv, MLRI_difcriRv, -1, VK3);
	filter2D(MLRI_criBv, MLRI_difcriBv, -1, VK3);
	RI_difcriGrh = abs(RI_difcriGrh);
	RI_difcriGbh = abs(RI_difcriGbh);
	RI_difcriRh = abs(RI_difcriRh);
	RI_difcriBh = abs(RI_difcriBh);
	RI_difcriGrv = abs(RI_difcriGrv);
	RI_difcriGbv = abs(RI_difcriGbv);
	RI_difcriRv = abs(RI_difcriRv);
	RI_difcriBv = abs(RI_difcriBv);
	MLRI_difcriGrh = abs(MLRI_difcriGrh);
	MLRI_difcriGbh = abs(MLRI_difcriGbh);
	MLRI_difcriRh = abs(MLRI_difcriRh);
	MLRI_difcriBh = abs(MLRI_difcriBh);
	MLRI_difcriGrv = abs(MLRI_difcriGrv);
	MLRI_difcriGbv = abs(MLRI_difcriGbv);
	MLRI_difcriRv = abs(MLRI_difcriRv);
	MLRI_difcriBv = abs(MLRI_difcriBv);
	// absolute value of iteration criteria
	RI_criGrh = abs(RI_criGrh);
	RI_criGbh = abs(RI_criGbh);
	RI_criRh = abs(RI_criRh);
	RI_criBh = abs(RI_criBh);
	RI_criGrv = abs(RI_criGrv);
	RI_criGbv = abs(RI_criGbv);
	RI_criRv = abs(RI_criRv);
	RI_criBv = abs(RI_criBv);
	MLRI_criGrh = abs(MLRI_criGrh);
	MLRI_criGbh = abs(MLRI_criGbh);
	MLRI_criRh = abs(MLRI_criRh);
	MLRI_criBh = abs(MLRI_criBh);
	MLRI_criGrv = abs(MLRI_criGrv);
	MLRI_criGbv = abs(MLRI_criGbv);
	MLRI_criRv = abs(MLRI_criRv);
	MLRI_criBv = abs(MLRI_criBv);

	// add Grand R(Gb and B) criteria residuals
	Mat RI_criGRh = (RI_criGrh + RI_criRh).mul(Mrh);
	Mat RI_criGBh = (RI_criGbh + RI_criBh).mul(Mbh);
	Mat RI_criGRv = (RI_criGrv + RI_criRv).mul(Mrv);
	Mat RI_criGBv = (RI_criGbv + RI_criBv).mul(Mbv);
	Mat MLRI_criGRh = (MLRI_criGrh + MLRI_criRh).mul(Mrh);
	Mat MLRI_criGBh = (MLRI_criGbh + MLRI_criBh).mul(Mbh);
	Mat MLRI_criGRv = (MLRI_criGrv + MLRI_criRv).mul(Mrv);
	Mat MLRI_criGBv = (MLRI_criGbv + MLRI_criBv).mul(Mbv);

	// add Grand R(Gb and B) gradient of criteria residuals
	Mat RI_difcriGRh = (RI_difcriGrh + RI_difcriRh).mul(Mrh);
	Mat RI_difcriGBh = (RI_difcriGbh + RI_difcriBh).mul(Mbh);
	Mat RI_difcriGRv = (RI_difcriGrv + RI_difcriRv).mul(Mrv);
	Mat RI_difcriGBv = (RI_difcriGbv + RI_difcriBv).mul(Mbv);
	Mat MLRI_difcriGRh = (MLRI_difcriGrh + MLRI_difcriRh).mul(Mrh);
	Mat MLRI_difcriGBh = (MLRI_difcriGbh + MLRI_difcriBh).mul(Mbh);
	Mat MLRI_difcriGRv = (MLRI_difcriGrv + MLRI_difcriRv).mul(Mrv);
	Mat MLRI_difcriGBv = (MLRI_difcriGbv + MLRI_difcriBv).mul(Mbv);

	// directional map of iteration criteria
	Mat RI_crih = RI_criGRh + RI_criGBh;
	Mat RI_criv = RI_criGRv + RI_criGBv;
	Mat MLRI_crih = MLRI_criGRh + MLRI_criGBh;
	Mat MLRI_criv = MLRI_criGRv + MLRI_criGBv;

	// directional gradient map of iteration criteria
	Mat RI_difcrih = RI_difcriGRh + RI_difcriGBh;
	Mat RI_difcriv = RI_difcriGRv + RI_difcriGBv;
	Mat MLRI_difcrih = MLRI_difcriGRh + MLRI_difcriGBh;
	Mat MLRI_difcriv = MLRI_difcriGRv + MLRI_difcriGBv;
	
	// Smoothing of Iteration Criteria 
	int sig = 2;
	Mat weightedF = getGaussianKernel(9, sigma, CV_32F);
	//Release the rest
	

	// Combine Vertical and Horizontal Color Differences
	// color difference gradient
	float difkernel[3] = { 1, 0, -1 };
	Mat dif_H_K(1, 3, CV_32F, difkernel);
	Mat dif_V_K(3, 1, CV_32F, difkernel);
	Mat V_diff_gradient, H_diff_gradient;
	filter2D(dif_H, H_diff_gradient, -1, dif_H_K);
	filter2D(dif_V, V_diff_gradient, -1, dif_V_K);

	H_diff_gradient = cv::abs(H_diff_gradient);
	V_diff_gradient = cv::abs(V_diff_gradient);

	// Directional weight
	Mat K = Mat::ones(5, 5, CV_32F);
	Mat hWeightSum, vWeightSum;
	filter2D(H_diff_gradient, hWeightSum, -1, K);
	filter2D(V_diff_gradient, vWeightSum, -1, K);
	Mat Wkernel = (Mat_<float>(1, 5) << 1, 0, 0, 0, 0);
	Mat Ekernel = (Mat_<float>(1, 5) << 0, 0, 0, 0, 1);
	Mat Nkernel = (Mat_<float>(5, 1) << 1, 0, 0, 0, 0);
	Mat Skernel = (Mat_<float>(5, 1) << 0, 0, 0, 0, 1);
	Mat wWeight, eWeight, nWeight, sWeight;
	filter2D(hWeightSum, wWeight, -1, Wkernel);
	filter2D(hWeightSum, eWeight, -1, Ekernel);
	filter2D(vWeightSum, nWeight, -1, Nkernel);
	filter2D(vWeightSum, sWeight, -1, Skernel);
	divide(1.0, (wWeight.mul(wWeight) + 1e-32), wWeight);
	divide(1.0, (eWeight.mul(eWeight) + 1e-32), eWeight);
	divide(1.0, (nWeight.mul(nWeight) + 1e-32), nWeight);
	divide(1.0, (sWeight.mul(sWeight) + 1e-32), sWeight);

	// combine directional color differences
	Mat weightedF = getGaussianKernel(9, sigma, CV_32F);
	Nkernel = (Mat_<float>(9, 1) << 1, 1, 1, 1, 1, 0, 0, 0, 0);
	Nkernel = Nkernel.mul(weightedF);
	float s = sum(Nkernel)[0];
	Nkernel /= s;
	Skernel = (Mat_<float>(9, 1) << 0, 0, 0, 0, 1, 1, 1, 1, 1);
	Skernel = Skernel.mul(weightedF) / s;
	transpose(Nkernel, Wkernel);
	transpose(Skernel, Ekernel);
	Mat fMulGradientSum_N, fMulGradientSum_S, fMulGradientSum_W, fMulGradientSum_E;
	filter2D(dif_V, fMulGradientSum_N, -1, Nkernel);
	filter2D(dif_V, fMulGradientSum_S, -1, Skernel);
	filter2D(dif_H, fMulGradientSum_W, -1, Wkernel);
	filter2D(dif_H, fMulGradientSum_E, -1, Ekernel);
	Mat totalWeight = nWeight + eWeight + wWeight + sWeight;
	Mat diff;
	divide(nWeight.mul(fMulGradientSum_N) + sWeight.mul(fMulGradientSum_S) + wWeight.mul(fMulGradientSum_W) + eWeight.mul(fMulGradientSum_E), totalWeight, diff);//(InputArray src1, InputArray src2, OutputArray dst, float scale=1, int dtype=-1)

	// Calculate Green by adding bayer raw data 
	finalBGR[1] = diff + Src1ch; //raw CFA data
	Mat imask = (mask[1] == 0); //[0, 1, 1] -> [255, 0, 0]
	imask.convertTo(imask, CV_32F, 1.0 / 255.0);
	finalBGR[1] = finalBGR[1].mul(imask) + bgr[1];

	// clip to 0~1

	threshold(finalBGR[1], finalBGR[1], 1.0, 1.0, THRESH_TRUNC); // > 1 to 1
	threshold(finalBGR[1], finalBGR[1], 0.0, 0.0, THRESH_TOZERO); // < 0 to 0

	// === 2.Red and Blue ===
	h = 5; //horizontal
	v = 5; //vertical
	eps = 0.0;

	// R interpolation
	Mat laplacian = (Mat_<float>(5, 5) << 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 4, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0);
	Mat lap_red, lap_green1;
	filter2D(bgr[2], lap_red, -1, laplacian);
	filter2D(finalBGR[1], lap_green1, -1, laplacian);
	Mat tentativeR = guided_filter_mlri(finalBGR[1], bgr[2], mask[2], lap_green1, lap_red, mask[2], h, v, eps);//G,R,mask,I,P,h,v,eps
	Mat residualR = (bgr[2] - tentativeR).mul(mask[2]);
	Mat bilinearKernel = (Mat_<float>(3, 3) << 0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25);
	filter2D(residualR, residualR, -1, bilinearKernel);
	finalBGR[2] = residualR + tentativeR;

	// B interpolation
	Mat lap_blue, lap_green2;
	filter2D(bgr[0], lap_blue, -1, laplacian);
	filter2D(finalBGR[1], lap_green2, -1, laplacian);
	Mat tentativeB = guided_filter_mlri(finalBGR[1], bgr[0], mask[0], lap_green2, lap_blue, mask[0], h, v, eps);
	Mat residualB = (bgr[0] - tentativeB).mul(mask[0]);
	filter2D(residualB, residualB, -1, bilinearKernel);
	finalBGR[0] = residualB + tentativeB;

	// Merge to single 3 channel Img
	merge(finalBGR, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
}