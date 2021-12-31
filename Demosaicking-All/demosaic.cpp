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
//I found this code online, apparently it can change patterns from grbg to rggb
void Demosaic(cv::Mat& img, cv::Mat& Dst, int BayerPatternFlag,int methodflag) {
	Mat BayerImage = img.clone();

	switch (BayerPatternFlag) { //turn all to RGGB by copyMakeBorder
	case 1: //RGGB

		break;
	case 2: //GRBG
		// R|GR|G
		// G|BG|B
		copyMakeBorder(BayerImage, BayerImage, 0, 0, 1, 1, cv::BORDER_REFLECT_101);
		break;
	case 3: //BGGR
		// R|GR|G
		// ------
		// G|BG|B
		// R|GR|G
		// ------
		// G|BG|B
		copyMakeBorder(BayerImage, BayerImage, 1, 1, 1, 1, cv::BORDER_REFLECT_101);
		break;
	case 4: //GBRG
		// RG
		// --
		// GB
		// RG
		// --
		// GB
		copyMakeBorder(BayerImage, BayerImage, 1, 1, 0, 0, cv::BORDER_REFLECT_101);
		break;
	default:
		std::cerr << "Wrong Bayer Pattern Flag" << endl;
		return;
	}
	switch (methodflag) {
	case 1: //Gradient Based Threshold Free
		demosaic_gbtf(BayerImage, Dst);
		break;
	case 2: //Residual Interpolation
		demosaic_residual(BayerImage, Dst);
		break;
	case 3: //MLRI
		demosaic_mlri(BayerImage, Dst);
		break;
	case 4: //MLRI + Weightage
		demosaic_tipri(BayerImage, Dst);
		break;
	//case 5: //ARI
	//	demosaic_ari(BayerImage, Dst);
	case 6: //NONA
		//demosaic_nona(BayerImage, Dst);
		break;
	default:
		std::cerr << "Wrong Method Pattern Flag" << endl;
		return;
	}
	//Rect (x, y, width, height);
	switch (BayerPatternFlag) { //turn all back from RGGB to Dst
	case 1: //RGGB
		break;
	case 2: //GRBG
		// R|GR|G
		// G|BG|B
		Dst = Dst(Rect(1, 0, BayerImage.cols - 2, BayerImage.rows));
		break;
	case 3: //BGGR
		// R|GR|G
		// ------
		// G|BG|B
		// R|GR|G
		// ------
		// G|BG|B
		Dst = Dst(Rect(1, 1, BayerImage.cols - 2, BayerImage.rows - 2));
		break;
	case 4: //GBRG
		// RG
		// --
		// GB
		// RG
		// --
		// GB
		Dst = Dst(Rect(0, 1, BayerImage.cols, BayerImage.rows - 2));
		break;
	}

	return;
}