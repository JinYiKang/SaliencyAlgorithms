#include "precomp.h"
//****************************************
//use function :HC_algor(cv::InputArray src, cv::OutputArray dst) to get a saliency map
//****************************************
//implementation HC algoithm of paper:
//Global Contrast based Salient Region Detection
//writen by: Ming-Ming Cheng, Niloy J. Mitra, Xiaolei Huang, Philip H. S. Torr, and Shi-Min Hu
//****************************************
//implemented by Jin Yikang 2018/11/2

float gamma(float x) {
	return x > 0.04045f ? std::powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92f);
}

float funcF(float x) {
	const float param1 = 0.008856f;
	const float param2 = 7.7870370f;
	const float param3 = 0.1379310f;

	return x > param1 ? std::powf(x, 1.f / 3.f) : (param2 * x + param3);
}

Vec3f bgr2lab(Vec3i bgr, float base) {
	const float param1 = 0.008856f;

	float r = gamma(bgr[2] / base);
	float g = gamma(bgr[1] / base);
	float b = gamma(bgr[0] / base);

	float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
	float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
	float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

	x /= 0.95047f;
	y /= 1.f;
	z /= 1.08883f;

	float L = y > param1 ? (116.f * funcF(y) - 16.f) : (903.3f*y);
	float A = 500.f*(funcF(x) - funcF(y));
	float B = 200.f*(funcF(y) - funcF(z));

	return Vec3f(L, A, B);
}

struct node {
	float val, saliency;
	Vec3i pos, inherit;

	node(float _v, int x, int y, int z) :
		val(_v), pos(Vec3i(x, y, z)), inherit(Vec3i()) ,saliency(0){}

	node() :val(0.f), saliency(0.f), pos(Vec3i()), inherit(Vec3i()) {}
};

struct dist {
	float dis;
	Vec3i pos;

	dist(float _d, Vec3i& _c) :dis(_d), pos(_c) {}

	dist() :dis(0.f), pos(Vec3i()) {}
};

class disList {
private:
	vector<dist> list;
public:
	disList(int _m) { list.reserve(_m); }

	void push(dist& _d) { list.push_back(_d); }

	float neighborDistSum(int m) {
		std::sort(list.begin(), list.end(),
			[](dist& d1, dist& d2)->bool {return d1.dis < d2.dis; });

		return std::accumulate(list.begin(), list.begin() + m, 0.f,
			[](float& a, dist& d) {return a + d.dis; });
	}
	
	dist& operator[](int idx) { return list[idx]; }
};

void histBasedSaliency(const Mat& src, Mat& map) {
	CV_Assert(src.type() == CV_8UC3 || src.type() == CV_16SC3);

	//get a 3 dimension hist
	const int channels[] = { 0,1,2 };
	int bbins, gbins, rbins;
	bbins = gbins = rbins = 12;
	const int histSize[] = { bbins,gbins,rbins };
	float r[] = { 0,256 };
	float g[] = { 0,256 };
	float b[] = { 0,256 };
	const float* ranges[] = { b,g,r };
	Mat _hist;
	cv::calcHist(&src, 1, channels, Mat(), _hist, 3, histSize, ranges);

	int allPixels = src.rows*src.cols;

	//get a color node vector
	vector<node> VN;
	for (int b = 0; b < bbins; ++b) {
		for (int g = 0; g < gbins; ++g) {
			for (int r = 0; r < rbins; ++r) {
				float v = _hist.at<float>(b, g, r) / (float)allPixels;
				if (v) {
					VN.push_back(node(v, b, g, r));
				}
			}
		}
	}

	//sort vector
	std::sort(VN.begin(), VN.end(),
		[](node& n1, node& n2)->bool {return n1.val > n2.val; }
	);

	//get those 95% color' position
	float sum = 0.f;
	typedef vector<node>::iterator VI;
	VI pt = std::find_if(VN.begin(), VN.end(),
		[&sum](node& n)->bool {sum += n.val; return sum >= 0.95f; }
	);

	//help those 5% color to inherit those 95% color
	VI ptmp = pt;
	while (ptmp != VN.end()) {
		Vec3i pos = ptmp->pos;
		VI fatherPt = std::min_element(VN.begin(), pt,
			[pos](node& n1, node& n2)->bool {return cv::norm(n1.pos, pos) < cv::norm(n2.pos, pos); }
		);
		fatherPt->val += ptmp->val;
		ptmp->inherit = fatherPt->pos;
		++ptmp;
	}

	//collect distance of each color with other colors and calc each color' salinency value.
	
	int colors = pt - VN.begin();
	vector<disList> VDL(colors, disList(colors / 4));
	ptmp = VN.begin();
	while (ptmp != pt) {
		VI ptmp2 = ptmp + 1;
		Vec3f lab1 = bgr2lab(ptmp->pos, 12.f);
		while (ptmp2 != pt) {
			Vec3f lab2 = bgr2lab(ptmp2->pos, 12.f);

			float labDist = (float)cv::norm(lab1, lab2);
			ptmp->saliency += labDist * (ptmp2->val);
			ptmp2->saliency += labDist * (ptmp->val);

			dist d1(labDist, ptmp2->pos);
			dist d2(labDist, ptmp->pos);
			VDL[ptmp - VN.begin()].push(d1);
			VDL[ptmp2 - VN.begin()].push(d2);

			++ptmp2;
		}
		++ptmp;
	}

	//get back project hist
	std::for_each(VN.begin(), pt, [&_hist](node& n) ->void {_hist.at<float>(n.pos) = n.saliency; });

	//smooth ,recomputer each color' saliency value.
	
	int m = colors / 4;
	for (int i = 0; i < colors; ++i) {
		float T = VDL[i].neighborDistSum(m);
		float newSaliency = 0.f;
		for (int j = 0; j < m; ++j) {
			newSaliency += (T - VDL[i][j].dis)*_hist.at<float>(VDL[i][j].pos);
		}
		newSaliency /= (m - 1)*T;
		VN[i].saliency = newSaliency;
	}
	
	//get back project hist
	std::for_each(VN.begin(), pt, [&_hist](node& n) ->void {_hist.at<float>(n.pos) = n.saliency; });
	std::for_each(pt, VN.end(), [&_hist](node& n)->void {_hist.at<float>(n.pos) = _hist.at<float>(n.inherit); });

	//back project
	Mat _src;
	src.convertTo(_src, CV_32FC3);
	cv::calcBackProject(&_src, 1, channels, _hist, map, ranges);

	//for imshow
	cv::normalize(map, map, 0, 1, CV_MINMAX);
}

void HC_algor(cv::InputArray src, cv::OutputArray dst) {
	Mat _src = src.getMat();
	dst.create(src.size(), CV_32FC1);
	Mat _dst = dst.getMat();
	histBasedSaliency(_src, _dst);
}