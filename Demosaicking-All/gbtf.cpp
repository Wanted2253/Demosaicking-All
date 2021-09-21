#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

void bayer_split(cv::Mat& Bayer, cv::Mat& Dst) {
	if (Bayer.channels() != 1) {
		std::cerr << "bayer_split allow only 1 channel raw bayer image " << std::endl;
		return;
	}
	Dst = cv::Mat::zeros(Bayer.rows, Bayer.cols, CV_8UC3);
	int channelNum;

	for (int row = 0; row < Bayer.rows; row++) {
		for (int col = 0; col < Bayer.cols; col++) {
			if (row % 2 == 0) { 
				channelNum = (col % 2 == 0) ? 2 : 1;
			}
			else {
				channelNum = (col % 2 == 0) ? 1 : 0;
			}
			Dst.at<Vec3b>(row, col).val[channelNum] = Bayer.at<uchar>(row, col);
		}
	}
	return;
}
void demosaic_gbtf(cv::Mat& Bayer, cv::Mat& Dst) {
	cv::Mat Src = Bayer.clone();
	if (Bayer.channels() == 1) {
		bayer_split(Bayer, Src);
	}


	Src.convertTo(Src, CV_32F, 1.0 / 255.0); 
	vector<Mat> bgr(3);
	vector<Mat> finalBGR(3);
	split(Src, bgr);
	split(Src, finalBGR);

	// 2.1. Green Channel Interpolation
	// Hamilton and Adams’ interpolation for B', G', R'
	float G_H, G_V;
	float R_H, R_V;
	float B_H, B_V;
	// for interpolation purpose
	copyMakeBorder(bgr[0], bgr[0], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	copyMakeBorder(bgr[1], bgr[1], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	copyMakeBorder(bgr[2], bgr[2], 2, 2, 2, 2, cv::BORDER_DEFAULT);
	// horizontal and vertical color difference
	Mat V_Diff(Src.size(), CV_32F, cv::Scalar(0));
	Mat H_Diff(Src.size(), CV_32F, cv::Scalar(0));
	// [Bayer]		[Horizontal]			[Vertical]
	// R G R G		D_gr D_gr D_gr D_gr		D_gr D_gb D_gr D_gb
	// G B G B  ->	D_gb D_gb D_gb D_gb  &	D_gr D_gb D_gr D_gb
	// R G R G		D_gr D_gr D_gr D_gr		D_gr D_gb D_gr D_gb
	// G B G B		D_gb D_gb D_gb D_gb		D_gr D_gb D_gr D_gb
	for (int row = 0; row < Src.rows; row++) {
		for (int col = 0; col < Src.cols; col++) {
			int i = row + 2;
			int j = col + 2;
			if (row % 2 == 0 && col % 2 == 0) { //Red
				G_H = (bgr[1].at<float>(i, j - 1) + bgr[1].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[2].at<float>(i, j) - bgr[2].at<float>(i, j - 2) - bgr[2].at<float>(i, j + 2)) * 0.25;
				G_V = (bgr[1].at<float>(i - 1, j) + bgr[1].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[2].at<float>(i, j) - bgr[2].at<float>(i - 2, j) - bgr[2].at<float>(i + 2, j)) * 0.25;

				// cal G' - R difference
				V_Diff.at<float>(row, col) = G_V - bgr[2].at<float>(i, j);
				H_Diff.at<float>(row, col) = G_H - bgr[2].at<float>(i, j);
			}
			else if (row % 2 == 1 && col % 2 == 1) { //Blue
				G_H = (bgr[1].at<float>(i, j - 1) + bgr[1].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[0].at<float>(i, j) - bgr[0].at<float>(i, j - 2) - bgr[0].at<float>(i, j + 2)) * 0.25;
				G_V = (bgr[1].at<float>(i - 1, j) + bgr[1].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[0].at<float>(i, j) - bgr[0].at<float>(i - 2, j) - bgr[0].at<float>(i + 2, j)) * 0.25;

				// cal G' - B difference
				V_Diff.at<float>(row, col) = G_V - bgr[0].at<float>(i, j);
				H_Diff.at<float>(row, col) = G_H - bgr[0].at<float>(i, j);
			}
			else if (row % 2 == 1 && col % 2 == 0) { //Green with Red Vertical / Blue Horizontal
				R_V = (bgr[2].at<float>(i - 1, j) + bgr[2].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i - 2, j) - bgr[1].at<float>(i + 2, j)) * 0.25;
				B_H = (bgr[0].at<float>(i, j - 1) + bgr[0].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i, j - 2) - bgr[1].at<float>(i, j + 2)) * 0.25;

				// cal G - R', G - B' difference
				V_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - R_V;
				H_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - B_H;
			}
			else { //Green with Red Horizontal / Blue Vertical
				R_H = (bgr[2].at<float>(i, j - 1) + bgr[2].at<float>(i, j + 1)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i, j - 2) - bgr[1].at<float>(i, j + 2)) * 0.25;
				B_V = (bgr[0].at<float>(i - 1, j) + bgr[0].at<float>(i + 1, j)) * 0.5
					+ (2 * bgr[1].at<float>(i, j) - bgr[1].at<float>(i - 2, j) - bgr[1].at<float>(i + 2, j)) * 0.25;

				// cal G - R', G - B' difference
				H_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - R_H;
				V_Diff.at<float>(row, col) = bgr[1].at<float>(i, j) - B_V;
			}
		}
	}

	// Final difference estimation for the target pixel
	copyMakeBorder(V_Diff, V_Diff, 4, 4, 4, 4, cv::BORDER_DEFAULT);
	copyMakeBorder(H_Diff, H_Diff, 4, 4, 4, 4, cv::BORDER_DEFAULT);
	Mat gr_Diff(Src.size(), CV_32F, cv::Scalar(0));
	Mat gb_Diff(Src.size(), CV_32F, cv::Scalar(0));
	Mat V_diff_gradient;
	Mat H_diff_gradient;
	float VHkernel[3] = { -1,0,1 };
	cv::Mat HK(1, 3, CV_32F, VHkernel);
	cv::Mat VK(3, 1, CV_32F, VHkernel);
	cv::filter2D(V_Diff, V_diff_gradient, -1, VK);
	cv::filter2D(H_Diff, H_diff_gradient, -1, HK);
	V_diff_gradient = cv::abs(V_diff_gradient); 
	H_diff_gradient = cv::abs(H_diff_gradient);

	float Weight[4];
	int startPoint[4][2] = { {-4, -2}, {0, -2}, {-2, -4}, {-2, 0} };
	float W_total;
	int a, b;
	for (int row = 0; row < Src.rows; row++) {
		int col = 0;
		if (row % 2 == 1) {
			col = 1;
		}
		for (; col < Src.cols; col += 2) {// Quincux pattern of G

			Weight[0] = Weight[1] = Weight[2] = Weight[3] = W_total = 0.0;
			for (int dir = 0; dir < 4; dir++) {
				a = startPoint[dir][0];
				b = startPoint[dir][1];
				for (int i = 0; i < 5; i++) {
					for (int j = 0; j < 5; j++) {
						if (dir < 2) { // N, S -> Vertical
							Weight[dir] += V_diff_gradient.at<float>(4 + row + a + i, 4 + col + b + j); // shift 4 due to copyMakeBorder
						}
						else { // W, E -> Horizontal
							Weight[dir] += H_diff_gradient.at<float>(4 + row + a + i, 4 + col + b + j);
						}
					}
				}
				Weight[dir] *= Weight[dir];
				Weight[dir] = 1.0 / Weight[dir];
				W_total += Weight[dir];
			}

			// calculate gr_Diff/gb_Diff & finalGreen 
			int i = row + 4;
			int j = col + 4;
			if (row % 2 == 0) { //Green @ Red
				gr_Diff.at<float>(row, col) =
					(Weight[0] * 0.2 * (V_Diff.at<float>(i - 4, j) + V_Diff.at<float>(i - 3, j) + V_Diff.at<float>(i - 2, j) + V_Diff.at<float>(i - 1, j) + V_Diff.at<float>(i, j))
						+ Weight[1] * 0.2 * (V_Diff.at<float>(i, j) + V_Diff.at<float>(i + 1, j) + V_Diff.at<float>(i + 2, j) + V_Diff.at<float>(i + 3, j) + V_Diff.at<float>(i + 4, j))
						+ Weight[2] * 0.2 * (H_Diff.at<float>(i, j - 4) + H_Diff.at<float>(i, j - 3) + H_Diff.at<float>(i, j - 2) + H_Diff.at<float>(i, j - 1) + H_Diff.at<float>(i, j))
						+ Weight[3] * 0.2 * (H_Diff.at<float>(i, j) + H_Diff.at<float>(i, j + 1) + H_Diff.at<float>(i, j + 2) + H_Diff.at<float>(i, j + 3) + H_Diff.at<float>(i, j + 4))
						) / W_total;

				finalBGR[1].at<float>(row, col) = finalBGR[2].at<float>(row, col) + gr_Diff.at<float>(row, col); // R + gb_Diff
			}
			else { //Green @ Blue
				gb_Diff.at<float>(row, col) =
					(Weight[0] * 0.2 * (V_Diff.at<float>(i - 4, j) + V_Diff.at<float>(i - 3, j) + V_Diff.at<float>(i - 2, j) + V_Diff.at<float>(i - 1, j) + V_Diff.at<float>(i, j))
						+ Weight[1] * 0.2 * (V_Diff.at<float>(i, j) + V_Diff.at<float>(i + 1, j) + V_Diff.at<float>(i + 2, j) + V_Diff.at<float>(i + 3, j) + V_Diff.at<float>(i + 4, j))
						+ Weight[2] * 0.2 * (H_Diff.at<float>(i, j - 4) + H_Diff.at<float>(i, j - 3) + H_Diff.at<float>(i, j - 2) + H_Diff.at<float>(i, j - 1) + H_Diff.at<float>(i, j))
						+ Weight[3] * 0.2 * (H_Diff.at<float>(i, j) + H_Diff.at<float>(i, j + 1) + H_Diff.at<float>(i, j + 2) + H_Diff.at<float>(i, j + 3) + H_Diff.at<float>(i, j + 4))
						) / W_total;

				finalBGR[1].at<float>(row, col) = finalBGR[0].at<float>(row, col) + gb_Diff.at<float>(row, col); // B + gb_Diff
			}
		}
	}

	// 2.2. Red and Blue Channel Interpolation
	float PrbData[49] = {
		0, 0, -0.03125, 0, -0.03125, 0, 0,
		0,0,0,0,0,0,0,
		-0.03125,0,0.3125,0,0.3125,0,-0.03125,
		0,0,0,0,0,0,0,
		-0.03125,0,0.3125,0,0.3125,0,-0.03125,
		0,0,0,0,0,0,0,
		0,0,-0.03125,0,-0.03125,0,0
	};
	cv::Mat Prb(7, 7, CV_32FC1, PrbData);

	copyMakeBorder(gr_Diff, gr_Diff, 3, 3, 3, 3, cv::BORDER_DEFAULT);
	copyMakeBorder(gb_Diff, gb_Diff, 3, 3, 3, 3, cv::BORDER_DEFAULT);
	// Red pixel values at blue locations and blue pixel values at redlocations
	// R G
	// G B
	for (int row = 0; row < Bayer.rows; row++) {
		int col = 0;
		if (row % 2 == 1) {
			col = 1;
		}

		for (; col < Bayer.cols; col += 2) {
			if (row % 2 == 0) { //Red
				//Blue @ Red
				finalBGR[0].at<float>(row, col) = finalBGR[1].at<float>(row, col) - cv::sum(gb_Diff(cv::Range(3 + row - 3, 3 + row + 3 + 1), cv::Range(3 + col - 3, 3 + col + 3 + 1)).mul(Prb))[0];
			}
			else { //Blue
			   //Red @ Blue
				finalBGR[2].at<float>(row, col) = finalBGR[1].at<float>(row, col) - cv::sum(gr_Diff(cv::Range(3 + row - 3, 3 + row + 3 + 1), cv::Range(3 + col - 3, 3 + col + 3 + 1)).mul(Prb))[0];
			}
		}
	}

	// For red and blue pixels at green locations, we use bilinearinterpolation over the closest four neighbors
	// R G
	// G B
	copyMakeBorder(finalBGR[0], bgr[0], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	copyMakeBorder(finalBGR[1], bgr[1], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	copyMakeBorder(finalBGR[2], bgr[2], 1, 1, 1, 1, cv::BORDER_DEFAULT);
	for (int row = 0; row < Src.rows; row++) {
		int col = 0;
		if (row % 2 == 0) {
			col = 1;
		}
		for (; col < Src.cols; col += 2) { //Green
			int i = row + 1;
			int j = col + 1;
			// Red
			finalBGR[2].at<float>(row, col) = finalBGR[1].at<float>(row, col)
				- (bgr[1].at<float>(i - 1, j) - bgr[2].at<float>(i - 1, j)) / 4.0
				- (bgr[1].at<float>(i + 1, j) - bgr[2].at<float>(i + 1, j)) / 4.0
				- (bgr[1].at<float>(i, j - 1) - bgr[2].at<float>(i, j - 1)) / 4.0
				- (bgr[1].at<float>(i, j + 1) - bgr[2].at<float>(i, j + 1)) / 4.0;
			// Blue
			finalBGR[0].at<float>(row, col) = finalBGR[1].at<float>(row, col)
				- (bgr[1].at<float>(i - 1, j) - bgr[0].at<float>(i - 1, j)) / 4.0
				- (bgr[1].at<float>(i + 1, j) - bgr[0].at<float>(i + 1, j)) / 4.0
				- (bgr[1].at<float>(i, j - 1) - bgr[0].at<float>(i, j - 1)) / 4.0
				- (bgr[1].at<float>(i, j + 1) - bgr[0].at<float>(i, j + 1)) / 4.0;
		}
	}

	merge(finalBGR, Dst);
	Dst.convertTo(Dst, CV_8U, 255.0);
	return;
}