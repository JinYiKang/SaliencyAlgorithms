#include "precomp.h"
#include <set>
//***********************
//The code is a implement of work:
//Context-Aware Saliency Detection 
//writen by Stas Goferman ,Lihi Zelnik-Manor,Ayellet Tal
//***********************
//Implemented by Jin Yikang 2018/10/30

//************some class and strct**********
struct getPos_cmp {
	void operator()(cv::Point2f& p, const int* pos) const {
		p.x = pos[0] / 250.f;
		p.y = pos[1] / 250.f;
	}
};

class KsimilarPatches {
private:
	std::multiset<float> minset;
	int K;
public:
	KsimilarPatches() :K(64) {}
	KsimilarPatches(int& k):K(k){}

	void push(float& f);

	float saliencyValue() {
		return 1.f - expf(-1.f / (float)K*accumulate(minset.begin(), minset.end(),0.f));
	}
};

void KsimilarPatches::push(float& f) {
	if (minset.size() < K)
		minset.insert(f);
	else {
		if (f < *(--minset.end())) {
			minset.erase(--minset.end());
			minset.insert(f);
		}
	}
}

//*************some tool function**********
void getWindowPos(vector<int>& wp, int r, int c, int size, int rows, int cols) {
	int topr = max(0, r - size / 2);
	int botr = min(r + size / 2, rows - 1) + 1;
	int lefc = max(0, c - size / 2);
	int rigc = min(c + size / 2, cols - 1) + 1;
	wp.push_back(topr);
	wp.push_back(botr);
	wp.push_back(lefc);
	wp.push_back(rigc);
}

template<class T>
T getIntegrAver(const Mat& integr, const vector<int>& wp) {
	int num = (wp[1] - wp[0])*(wp[3] - wp[2]);
	return (integr.at<T>(wp[1], wp[3]) - integr.at<T>(wp[1], wp[2])
		- integr.at<T>(wp[0], wp[3]) + integr.at<T>(wp[0], wp[2])) / (float)num;
}

float distance_calu(const Vec3f& p_lab, const Vec3f& q_lab, const Vec2f& p_pos, const Vec2f& q_pos) {
	return (float)norm(p_lab, q_lab, NORM_L2) / (1.f + 3.f*(float)norm(p_pos, q_pos, NORM_L2));
}

//************search function*************
void RqCompare(const Mat& integr_lab, const Mat& integr_pos, KsimilarPatches& KSP,
	const vector<float>& Rq, const Vec3f& p_lab, const Vec2f& p_pos,
	int r, int c, int rows, int cols, int size, float R)
{
	for (int i = 0; i < Rq.size(); ++i) {
		if (size*R*Rq[i] < size*0.2f)
			break;

		vector<int> wp;
		getWindowPos(wp, r, c, cvRound(size*R*Rq[i]), rows, cols);
		Vec3f q_lab = getIntegrAver<Vec3f>(integr_lab, wp);
		Vec2f q_pos = getIntegrAver<Vec2f>(integr_pos, wp);
		float v = distance_calu(p_lab, q_lab, p_pos, q_pos);
		KSP.push(v);
	}
}

void backwardSearch(const Mat& integr_lab, const Mat& integr_pos, const vector<float>& Rq, KsimilarPatches& KSP,
	int r, int c, int size, float R, int rows, int cols, const Vec3f& p_lab, const Vec2f& p_pos)
{
	int ws = cvRound(size*R);
	for (int br = r; br >= 0; br -= ws / 2) {
		//left
		for (int bc = c - ws / 2; bc >= 0; bc -= ws / 2) {
			RqCompare(integr_lab, integr_pos, KSP, Rq, p_lab, p_pos, br, bc, rows, cols, size, R);
		}
		//right
		if (br == r) continue;

		for (int bc = c; bc < cols; bc += ws / 2) {
			RqCompare(integr_lab, integr_pos, KSP, Rq, p_lab, p_pos, br, bc, rows, cols, size, R);
		}
	}
}

void forewardSearch(const Mat& integr_lab, const Mat& integr_pos, const vector<float>& Rq, KsimilarPatches& KSP,
	int r, int c, int size, float R, int rows, int cols, const Vec3f& p_lab, const Vec2f& p_pos)
{
	int ws = cvRound(size*R);
	for (int fr = r; fr < rows; fr += ws / 2) {
		//right
		for (int fc = c + ws / 2; fc < cols; fc += ws / 2) {
			RqCompare(integr_lab, integr_pos, KSP, Rq, p_lab, p_pos, fr, fc, rows, cols, size, R);
		}
		//left
		if (fr == r) continue;

		for (int fc = c; fc >= 0; fc -= ws / 2) {
			RqCompare(integr_lab, integr_pos, KSP, Rq, p_lab, p_pos, fr, fc, rows, cols, size, R);
		}
	}
}

//************saliency core*************
void saliencyCore(const Mat& integr_lab, const Mat& integr_pos, Mat& map, int size, float R, const vector<float>& Rq) {
	cv::MatIterator_<float> pt = map.begin<float>();
	int step = map.cols;

	for (int i = 0; pt != map.end<float>(); ++i, ++pt) {
		int r = i / step;
		int c = i % step;

		vector<int> wp;
		KsimilarPatches KSP;
		getWindowPos(wp, r, c, cvRound(size*R), map.rows, map.cols);
		Vec3f p_lab = getIntegrAver<Vec3f>(integr_lab, wp);
		Vec2f p_pos = getIntegrAver<Vec2f>(integr_pos, wp);

		backwardSearch(integr_lab, integr_pos, Rq, KSP, r, c, size, R, map.rows, map.cols, p_lab, p_pos);
		forewardSearch(integr_lab, integr_pos, Rq, KSP, r, c, size, R, map.rows, map.cols, p_lab, p_pos);

		*pt += KSP.saliencyValue();
	}
}

//************function you can use**********
void contextAware(const Mat& src, Mat& map) {
	CV_Assert(src.type() == CV_8UC3 || src.type() == CV_16SC3);

	//resize the image to the size of 250 according to the paper
	Mat _src;
	if (src.rows > src.cols)
		resize(src, _src, Size(250 * src.cols / src.rows, 250));
	else
		resize(src, _src, Size(250, 250 * src.rows / src.cols));

	//convert to lab
	cv::cvtColor(_src, _src, COLOR_BGR2Lab);
	//normalize
	if (src.type() == CV_8UC3)
		_src.convertTo(_src, CV_32F, 1.0 / 255.0);
	else
		_src.convertTo(_src, CV_32F, 1.0 / 65535.0);
	
	//get pos mat and normalize 
	Mat pos_mat(_src.size(), CV_32FC2);
	pos_mat.forEach<Point2f>(getPos_cmp());

	//get integral image
	Mat integr_lab, integr_pos;
	cv::integral(_src, integr_lab, CV_32F);
	cv::integral(pos_mat, integr_pos, CV_32F);

	//get some data according to paper
	vector<float> R = { 1.f,0.8f,0.5f,0.3f };
	vector<float> Rq = { 1.f,0.5f,0.25f };

	//let the saliency core work
	Mat _map = Mat::zeros(_src.size(), CV_32FC1);
	for (int i = 0; i < R.size(); ++i) {
		saliencyCore(integr_lab, integr_pos, _map, 7, R[i], Rq);
	}

	//get a copy
	_map.copyTo(map);
}




