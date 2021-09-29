#include <cstdio>
#include <iostream>
#include <string>
#include <cstring>jjjjjjg
#include <vector>
#include <opencv2/opencv.hpp>
#include "demosaic.h"
using namespace std;
using namespace cv;
//Pattern: 1 RGGB, 2 GRBG, 3 BGGR, 4 GBRG
//Method: 1 GBTF, 2 RI,3 MLRI, 4 MLRI + Wei, 5 ARI
int main() {
	string path = "bayer_pattern_img.bmp";
	Mat img = imread(path);
	Mat dst_gbtf,dst_ri,dst_mlri;
	int pattern = 2;
	Demosaic(img, dst_gbtf, pattern, 1);
	Demosaic(img, dst_ri, pattern, 2);
	Demosaic(img, dst_mlri, pattern, 3);

	imshow("Output-GBTF.png", dst_gbtf);
	imshow("Output-RI.png", dst_ri);
	imshow("Output-MLRI.png", dst_mlri);
	/*double k = getpsnr(dst, img);
	cout << k;*/
	/*imwrite("Output-GBTF.png", dst_gbtf);
	imwrite("Output-RI.png", dst_ri);
	imwrite("Output-MLRI.png", dst_mlri);*/
	waitKey(0);
	return 0;
}