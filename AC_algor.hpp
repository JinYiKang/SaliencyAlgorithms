#include "precomp.h"
//*************************
//Use function : void AC_algorithms(const Mat& src, Mat& map, int R1_size = 1);
//to process image,if you want to imshow the fianl map you got after process,
//use cv::normalize with norm type :NORM_MINMAX.
//*************************
//The code is a implementation of work:
//Salient Region Detection and Segmentation 
//writen by Radhakrishna Achanta, Francisco Estrada, Patricia Wils, and Sabine SÄusstrunk
//****************************
//Implemented by Jin Yikang,2018/10/30

//use Euclidean distance
template<class T>
float distance_calu(const Vec<T, 3>& v1, const Vec3f& v2) {
	return sqrtf(
		powf((float)v1[0] - v2[0], 2) +
		powf((float)v1[1] - v2[1], 2) +
		powf((float)v1[2] - v2[2], 2)
	);
}

//when R1 size == 1,use the default scalingCore function
template<class T>
void scalingCore(const Mat& src, const Mat& integr, Mat& map, int R2_size) {
	cv::MatConstIterator_<T> pt1 = src.begin<T>();
	cv::MatIterator_<float> pt2 = map.begin<float>();
	int step = src.cols;

	for (int i = 0; pt1 != src.end<T>(); pt1++, pt2++, i++) {
		int r = i / step;
		int c = i % step;
		int topr = std::max(0, r - R2_size / 2);
		int botr = std::min(r + R2_size / 2, src.rows - 1) + 1;//plus 1
		int lefc = std::max(0, c - R2_size / 2);
		int rigc = std::min(c + R2_size / 2, src.cols - 1) + 1;//plus 1

		Vec3f v = integr.at<Vec3f>(botr, rigc) - integr.at<Vec3f>(botr, lefc)
			- integr.at<Vec3f>(topr, rigc) + integr.at<Vec3f>(topr, lefc);

		int num = (botr - topr)*(rigc - lefc);
		v /= num;

		*pt2 += distance_calu(*pt1, v);
	}
}

//when R1 size > 1,use another overload scalingCore function
void scalingCore(const Mat& integr, Mat& map, int R1_size, int R2_size) {
	cv::MatIterator_<float> pt = map.begin<float>();
	int step = map.cols;

	for (int i = 0; pt != map.end<float>(); pt++, i++) {
		int r = i / step;
		int c = i % step;
		int r1_topr = max(0, r - R1_size / 2),                 r2_topr = max(0, r - R2_size / 2);
		int r1_botr = min(r + R1_size / 2, map.rows - 1) + 1,  r2_botr = min(r + R2_size / 2, map.rows - 1) + 1;
		int r1_lefc = max(0, c - R1_size / 2),                 r2_lefc = max(0, c - R2_size / 2);
		int r1_rigc = min(c + R1_size / 2, map.cols - 1) + 1,  r2_rigc = min(c + R2_size / 2, map.cols - 1) + 1;

		Vec3f v1 = integr.at<Vec3f>(r1_botr, r1_rigc) - integr.at<Vec3f>(r1_botr, r1_lefc)
			- integr.at<Vec3f>(r1_topr, r1_rigc) + integr.at<Vec3f>(r1_topr, r1_lefc);

		int num = (r1_botr - r1_topr)*(r1_rigc - r1_lefc);
		v1 /= num;

		Vec3f v2 = integr.at<Vec3f>(r2_botr, r2_rigc) - integr.at<Vec3f>(r2_botr, r2_lefc)
			- integr.at<Vec3f>(r2_topr, r2_rigc) + integr.at<Vec3f>(r2_topr, r2_lefc);

		num = (r2_botr - r2_topr)*(r2_rigc - r2_lefc);
		v2 /= num;

		*pt += distance_calu(v1, v2);
	}
}

void ScalingLoop(const Mat& src, const Mat& integr, Mat& map, int R1_size, int R2_size) {
	const int loops = 3;

	if (R1_size == 1) {
		if (src.type() == CV_8UC3) {

			for (int i = 0; i < loops; i++) {
				if (R2_size <= R1_size) return;

				scalingCore<Vec<uchar,3>>(src, integr, map, R2_size);
				R2_size /= 2;
			}
		}
		else {
			for (int i = 0; i < loops; i++) {
				if (R2_size <= R1_size) return;

				scalingCore<Vec<short,3>>(src, integr, map, R2_size);
				R2_size /= 2;
			}
		}
	}
	else {
		for (int i = 0; i < loops; i++) {
			if (R2_size <= R1_size) return;

			scalingCore(integr, map, R1_size, R2_size);
			R2_size /= 2;
		}
	}
}

//default size of r1 = 1
void AC_algorithms(const Mat& src,Mat& dst, int R1_size = 1) {
	CV_Assert(src.type() == CV_8UC3 || src.type() == CV_16SC3);

	Mat _src;
	cv::cvtColor(src, _src, COLOR_BGR2Lab);

	//get integral image
	Mat integralImage;
	cv::integral(_src, integralImage,CV_32F);

	//get R2 size
	int R2_size = std::min(_src.rows, _src.cols) / 2;

	//scale 
	
	Mat _map = Mat::zeros(_src.size(), CV_32FC1);
	ScalingLoop(_src, integralImage, _map, R1_size, R2_size);

	//get a copy
	_map.copyTo(dst);
	cv::normalize(dst, dst, 0, 1, NORM_MINMAX);
}


