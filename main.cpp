#include "HC_algor.hpp"
#include "AC_algor.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
int main()
{
	Mat src = imread("F:/vs2015_code/20.jpg");
	Mat map;
	HC_algor(src, map);
	//AC_algorithms(src, map);
	//histBasedSaliency(src, map);
	imshow("map", map);
	waitKey(0);
}