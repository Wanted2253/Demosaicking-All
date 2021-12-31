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
Mat box_filter_modified(const Mat& I, int h, int v) {
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
Mat guided_filter_tipri(const Mat& originG, const Mat& originR, const Mat& mask, const Mat& originI, const Mat& originP, const Mat& M, int h = 2, int v = 2, float eps = 0.0) {
	bool flag = false;
	Mat p, I, G, R;
	G = originG.clone();
	R = originR.clone();
	p = originP.clone();
	I = originI.clone();

	Mat N = box_filter_modified(M, h, v);
	Mat temp = (N == 0);
	temp.convertTo(temp, CV_32F, 1.0 / 255.0);
	N = N + temp;
	// The size of each local patch; N=(2h+1)*(2v+1) except for boundary pixels.
	Mat N2 = box_filter_modified(Mat::ones(I.rows, I.cols, CV_32F), h, v);

	Mat mean_IP = box_filter_modified(I.mul(p).mul(M), h, v);
	divide(mean_IP, N, mean_IP);
	Mat mean_II = box_filter_modified(I.mul(I).mul(M), h, v);
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
	Mat N3 = box_filter_modified(mask, h, v);
	Mat temp1 = (N3 == 0);
	temp1.convertTo(temp1, CV_32F, 1.0 / 255.0);
	N3 = N3 + temp1;
	Mat mean_G = box_filter_modified(G.mul(mask), h, v);
	divide(mean_G, N3, mean_G);
	Mat mean_R = box_filter_modified(R.mul(mask), h, v);
	divide(mean_R, N3, mean_R);
	Mat b = mean_R - a.mul(mean_G);

	//Weighted Average
	Mat dif;
	Mat dif1 = a.mul(a).mul((box_filter_modified(G.mul(G).mul(mask), h, v)));
	Mat dif2 = b.mul(b).mul(N3);
	Mat dif3 = box_filter_modified(R.mul(R).mul(mask), h, v);
	Mat dif4 = a.mul(b).mul(box_filter_modified(G.mul(mask), h, v));
	dif4 *= 2;
	Mat dif5 = b.mul(box_filter_modified(R.mul(mask), h, v));
	dif5 = dif5 * 2;
	Mat dif6 = a.mul(box_filter_modified(R.mul(G).mul(mask), h, v));
	dif6 = dif6 * 2;
	dif = dif1 + dif2 + dif3 + dif4 - dif5 - dif6;
	divide(dif, N3, dif);
	for (int row = 0; row < dif.rows; row++) {
		float* p = dif.ptr<float>(row);
		for (int col = 0; col < dif.cols; col++) {
			if (flag == false) {
				cout << p[col] << endl;
				flag = true;
			}

		}
	}
	th = 0.01;
	for (int row = 0; row < dif.rows; row++) {
		float* p = dif.ptr<float>(row);
		for (int col = 0; col < dif.cols; col++) {
			if (p[col] < th) {
				p[col] = th;
			}
		}
	}
	divide(1.0, dif, dif);
	Mat wdif = box_filter_modified(dif, h, v);
	for (int row = 0; row < wdif.rows; row++) {
		float* p1 = wdif.ptr<float>(row);
		for (int col = 0; col < wdif.cols; col++) {
			if (p1[col] < th) {
				p1[col] = th;
			}
		}
	}
	//Mean a and Mean b
	Mat mean_a = box_filter_modified(a.mul(dif), h, v);
	divide(mean_a, wdif, mean_a);
	Mat mean_b = box_filter_modified(b.mul(dif), h, v);
	divide(mean_b, wdif, mean_b);
	// step: 5
	Mat q = mean_a.mul(G) + mean_b;

	return q;
}