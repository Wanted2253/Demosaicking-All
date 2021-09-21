#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "guidedfiltermlri.h"
using namespace std;
using namespace cv;
void toSingleChannelmlri(cv::Mat& src, cv::Mat& dst) { 
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
void bayer_splitmlri(cv::Mat& Bayer, cv::Mat& Dst) {
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
void bayer_maskmlri(cv::Mat& Bayer, cv::Mat& Dst) {
	Mat temp = cv::Mat::ones(Bayer.size(), CV_8U);
	bayer_splitmlri(temp, Dst);// or split(Bayer, temp)
}

void demosaic_mlri(cv::Mat& Bayer, cv::Mat& Dst, float sigma = 1.0) {
	cv::Mat Src = Bayer.clone();
	if (Bayer.channels() == 1) { //input 1 channel -> 3 channel Bayer
		bayer_splitmlri(Bayer, Src); // or split(Bayer, Src);
	}

	Mat Src1ch;
	toSingleChannelmlri(Src, Src1ch); // or merge(Src, Src1ch);
	Src1ch.convertTo(Src1ch, CV_32F, 1.0 / 255.0); //normalize

	Mat tempMask;
	bayer_maskmlri(Src, tempMask);
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

	// Creating guide images
	Mat GuideG_H = bgr[1] + rawH.mul(mask[2]) + rawH.mul(mask[0]);
	Mat GuideR_H = bgr[2] + rawH.mul(maskGr); 
	Mat GuideB_H = bgr[0] + rawH.mul(maskGb); 
	Mat GuideG_V = bgr[1] + rawV.mul(mask[2]) + rawV.mul(mask[0]);
	Mat GuideR_V = bgr[2] + rawV.mul(maskGb); 
	Mat GuideB_V = bgr[0] + rawV.mul(maskGr);

	// Tentative image
	int h = 5; //horizontal
	int v = 0; //vertical
	float eps = 0;
	//Horizontal
	float FHkernel[5] = { -1 , 0 , 2 , 0 , -1 };
	Mat FHK(1, 5, CV_32F, FHkernel);
	Mat diffr1H, diffGr1H;
	filter2D(bgr[2], diffr1H, -1, FHK);
	filter2D(GuideG_H.mul(mask[2]), diffGr1H, -1, FHK);
	Mat tentativeR_H = guided_filter_mlri(GuideG_H, bgr[2], mask[2], diffGr1H, diffr1H, mask[2], h, v, eps);//G,R,mask,I,P,M,h,v,eps
	Mat diffr2H, diffGr2H;
	filter2D(bgr[1].mul(maskGr), diffGr2H, -1, FHK);
	filter2D(GuideR_H.mul(maskGr), diffr2H, -1, FHK);
	Mat tentativeGr_H = guided_filter_mlri(GuideR_H, bgr[1].mul(maskGr), maskGr, diffr2H, diffGr2H, maskGr, h, v, eps);// need mul(mask) because Green has two location
	Mat diffb1H, diffGb1H;
	filter2D(bgr[0], diffb1H, -1, FHK);
	filter2D(GuideG_H.mul(mask[0]), diffGb1H, -1, FHK);
	Mat tentativeB_H = guided_filter_mlri(GuideG_H, bgr[0], mask[0], diffGb1H, diffb1H, mask[0], h, v, eps);
	Mat diffb2H, diffGb2H;
	filter2D(bgr[1].mul(maskGb), diffGb2H, -1, FHK);
	filter2D(GuideB_H.mul(maskGb), diffb2H, -1, FHK);
	Mat tentativeGb_H = guided_filter_mlri(GuideB_H, bgr[1].mul(maskGb), maskGb, diffb2H, diffGb2H, maskGb, h, v, eps);

	// vertical part
	float FVkernel[5] = { -1 , 0 , 2 , 0 , -1 };
	Mat FVK(5, 1, CV_32F, FVkernel);
	Mat diffr1V, diffGr1V;
	filter2D(bgr[2], diffr1V, -1, FVK);
	filter2D(GuideG_V.mul(mask[2]), diffGr1V, -1, FVK);
	Mat tentativeR_V = guided_filter_mlri(GuideG_V, bgr[2], mask[2], diffGr1V, diffr1V, mask[2], v, h, eps);//G,R,mask,I,P,M,h,v,eps
	Mat diffr2V, diffGr2V;
	filter2D(bgr[1].mul(maskGr), diffGr2V, -1, FVK);
	filter2D(GuideR_V.mul(maskGr), diffr2V, -1, FVK);
	Mat tentativeGr_V = guided_filter_mlri(GuideR_V, bgr[1].mul(maskGr), maskGr, diffr2V, diffGr2V, maskGr, v, h, eps);// need mul(mask) because Green has two location
	Mat diffb1V, diffGb1V;
	filter2D(bgr[0], diffb1V, -1, FVK);
	filter2D(GuideG_V.mul(mask[0]), diffGb1V, -1, FVK);
	Mat tentativeB_V = guided_filter_mlri(GuideG_V, bgr[0], mask[0], diffGb1V, diffb1V, mask[0], v, h, eps);
	Mat diffb2V, diffGb2V;
	filter2D(bgr[1].mul(maskGb), diffGb2V, -1, FVK);
	filter2D(GuideB_V.mul(maskGb), diffb2V, -1, FVK);
	Mat tentativeGb_V = guided_filter_mlri(GuideB_V, bgr[1].mul(maskGb), maskGb, diffb2V, diffGb2V, maskGb, v, h, eps);

	//release Guided Image
	GuideG_H.release();
	GuideR_H.release();
	GuideB_H.release();
	GuideG_V.release();
	GuideR_V.release();
	GuideB_V.release();

	// Residual
	Mat residualGr_H = (bgr[1] - tentativeGr_H).mul(maskGr);
	Mat residualGb_H = (bgr[1] - tentativeGb_H).mul(maskGb);
	Mat residualR_H = (bgr[2] - tentativeR_H).mul(mask[2]);
	Mat residualB_H = (bgr[0] - tentativeB_H).mul(mask[0]);
	Mat residualGr_V = (bgr[1] - tentativeGr_V).mul(maskGb);
	Mat residualGb_V = (bgr[1] - tentativeGb_V).mul(maskGr);
	Mat residualR_V = (bgr[2] - tentativeR_V).mul(mask[2]);
	Mat residualB_V = (bgr[0] - tentativeB_V).mul(mask[0]);

	// Residual interpolation
	filter2D(residualGr_H, residualGr_H, -1, HK); //original matlab using 'replicate'
	filter2D(residualGb_H, residualGb_H, -1, HK);
	filter2D(residualR_H, residualR_H, -1, HK);
	filter2D(residualB_H, residualB_H, -1, HK);
	// verical part
	filter2D(residualGr_V, residualGr_V, -1, VK);
	filter2D(residualGb_V, residualGb_V, -1, VK);
	filter2D(residualR_V, residualR_V, -1, VK);
	filter2D(residualB_V, residualB_V, -1, VK);

	// Add tentative image
	Mat Gr_H = (tentativeGr_H + residualGr_H).mul(mask[2]);
	Mat Gb_H = (tentativeGb_H + residualGb_H).mul(mask[0]);
	Mat R_H = (tentativeR_H + residualR_H).mul(maskGr);
	Mat B_H = (tentativeB_H + residualB_H).mul(maskGb);
	Mat Gr_V = (tentativeGr_V + residualGr_V).mul(mask[2]);
	Mat Gb_V = (tentativeGb_V + residualGb_V).mul(mask[0]);
	Mat R_V = (tentativeR_V + residualR_V).mul(maskGb);
	Mat B_V = (tentativeB_V + residualB_V).mul(maskGr);

	// Vertical and horizontal color difference 
	Mat dif_H = bgr[1] + Gr_H + Gb_H - bgr[2] - bgr[0] - R_H - B_H;
	Mat dif_V = bgr[1] + Gr_V + Gb_V - bgr[2] - bgr[0] - R_V - B_V;

	//Release the rest
	tentativeR_H.release();
	tentativeGr_H.release();
	tentativeGb_H.release();
	tentativeB_H.release();
	tentativeR_V.release();
	tentativeGr_V.release();
	tentativeGb_V.release();
	tentativeB_V.release();

	residualGr_H.release();
	residualGb_H.release();
	residualR_H.release();
	residualB_H.release();
	residualGr_V.release();
	residualGb_V.release();
	residualR_V.release();
	residualB_V.release();

	Gr_H.release();
	Gb_H.release();
	R_H.release();
	B_H.release();
	Gr_V.release();
	Gb_V.release();
	R_V.release();
	B_V.release();

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