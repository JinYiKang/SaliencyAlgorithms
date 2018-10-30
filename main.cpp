#include "AC_algor.hpp"
#include <opencv2/highgui.hpp>

int main()
{
	Mat src = imread("F:/vs2015_code/9.jpg");
	Mat map;
	AC_algorithms(src, map);
	normalize(map, map, 0, 1, NORM_MINMAX);
	return 0;
}