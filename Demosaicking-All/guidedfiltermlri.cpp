#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
// =============== Guided Filter Modified ===================
// I: input
// h, v: local window radius
Mat box_filter_modified_mlri(const Mat& I, int h, int v) {
	Mat result;
	blur(I, result, Size(2 * h + 1, 2 * v + 1)); //width * height
	return result * (2 * h + 1) * (2 * v + 1);
}
// G: Original Green Subsample
// R or B: Original Red or Blue Subsample
// p: origin img (assume float or CV_32F, 0.0 ~ 1.0)
// I: guided img
// M: binary mask
// h, v: local window radius
// eps: Regularization Parameter
Mat guided_filter_mlri(const Mat& originG, const Mat& originR, const Mat& mask, const Mat& originI, const Mat& originP, const Mat& M, int h = 2, int v = 2, float eps = 0.0) {
	Mat p, I, G, R;
	G = originG.clone();
	R = originR.clone();
	p = originP.clone();
	I = originI.clone();

	Mat N = box_filter_modified_mlri(M, h, v);
	Mat temp = (N == 0);
	temp.convertTo(temp, CV_32F, 1.0 / 255.0);
	N = N + temp;
	// The size of each local patch; N=(2h+1)*(2v+1) except for boundary pixels.
	Mat N2 = box_filter_modified_mlri(Mat::ones(I.rows, I.cols, CV_32F), h, v);

	Mat mean_IP = box_filter_modified_mlri(I.mul(p).mul(M), h, v);
	divide(mean_IP, N, mean_IP);
	Mat mean_II = box_filter_modified_mlri(I.mul(I).mul(M), h, v);
	divide(mean_II, N, mean_II);
	float th = 0.00001;
	for (int row = 0; row < mean_II.rows; row++) {
		float* p = mean_II.ptr<float>(row);
		for (int col = 0; col < mean_II.cols; col++) {
			if (p[col] < th) {
				p[col] = th;
			}
		}
	}
	Mat a;
	if (mean_II.channels() == 3) {
		a = mean_IP / (mean_II + Scalar(eps, eps, eps)); //otherwise only 1 channel get added
	}
	else {
		a = mean_IP / (mean_II + eps);
	}
	Mat N3 = box_filter_modified_mlri(mask, h, v);
	Mat temp1 = (N3 == 0);
	temp1.convertTo(temp1, CV_32F, 1.0 / 255.0);
	N3 = N3 + temp1;
	Mat mean_G = box_filter_modified_mlri(G.mul(mask), h, v);
	divide(mean_G, N3, mean_G);
	Mat mean_R = box_filter_modified_mlri(R.mul(mask), h, v);
	divide(mean_R, N3, mean_R);
	Mat b = mean_R - a.mul(mean_G);
	Mat mean_a = box_filter_modified_mlri(a, h, v);
	divide(mean_a, N2, mean_a);
	Mat mean_b = box_filter_modified_mlri(b, h, v);
	divide(mean_b, N2, mean_b);
	// step: 5
	Mat q = mean_a.mul(G) + mean_b;
	Mat res;

	//q.convertTo(q, CV_8U, 255.0);
	return q;
}