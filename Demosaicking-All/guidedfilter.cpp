#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

// =============== Guided Filter Modified ===================
// I: imput
// h, v: local window radius
Mat box_filter_modified_ri(const Mat& I, int h, int v) {
	Mat result;
	// = call boxFiltr (all average)
	// boxFilter(src, dst, src.type(), anchor, true, borderType).
	blur(I, result, Size(2 * h + 1, 2 * v + 1)); //width * height
	return result * (2 * h + 1) * (2 * v + 1);
}
// p: origin img (assume float or CV_32F, 0.0 ~ 1.0)
// I: guided img
// M: binary mask
// h, v: local window radius
// eps: regularization parameter (0.1)^2, (0.2)^2...
Mat guided_filter_modified(const Mat& originP, const Mat& originI, const Mat& M, int h = 2, int v = 2, float eps = 0.0) {
	Mat p, I;
	p = originP.clone();
	I = originI.clone();

	// seems need mul(M) for all I if I will not be zero outside mask

	//The number of the sampled pixels in each local patch
	Mat N = box_filter_modified_ri(M, h, v);
	Mat temp = (N == 0);
	temp.convertTo(temp, CV_32F, 1.0 / 255.0);
	N = N + temp;
	// The size of each local patch; N=(2h+1)*(2v+1) except for boundary pixels.
	Mat N2 = box_filter_modified_ri(Mat::ones(I.rows, I.cols, CV_32F), h, v);

	// step: 1
	Mat mean_I = box_filter_modified_ri(I.mul(M), h, v);
	divide(mean_I, N, mean_I);
	Mat mean_p = box_filter_modified_ri(p, h, v);
	divide(mean_p, N, mean_p);
	Mat corr_I = box_filter_modified_ri(I.mul(I).mul(M), h, v); //mul: element wise mul
	divide(corr_I, N, corr_I);
	Mat corr_Ip = box_filter_modified_ri(I.mul(p), h, v);
	divide(corr_Ip, N, corr_Ip);

	// step: 2
	Mat var_I = corr_I - mean_I.mul(mean_I);
	//threshold parameter
	float th = 0.00001;
	for (int row = 0; row < var_I.rows; row++) {
		float* p = var_I.ptr<float>(row);
		for (int col = 0; col < var_I.cols; col++) {
			if (p[col] < th) {
				p[col] = th;
			}
		}
	}

	Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);
	// step: 3
	Mat a;
	if (var_I.channels() == 3) {
		a = cov_Ip / (var_I + Scalar(eps, eps, eps)); //otherwise only 1 channel get added
	}
	else {
		a = cov_Ip / (var_I + eps);
	}
	Mat b = mean_p - a.mul(mean_I);
	// step: 4
	Mat mean_a = box_filter_modified_ri(a, h, v);
	divide(mean_a, N2, mean_a);
	Mat mean_b = box_filter_modified_ri(b, h, v);
	divide(mean_b, N2, mean_b);
	// step: 5
	Mat q = mean_a.mul(I) + mean_b;
	Mat res;

	//q.convertTo(q, CV_8U, 255.0);
	return q;
}