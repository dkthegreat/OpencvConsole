// OpencvConsole.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_flann2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_nonfree2413d.lib")
#pragma comment(lib, "opencv_calib3d2413d.lib")

#pragma comment(lib, "opencv_imgproc2413d.lib")

//#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
using namespace std;

using namespace cv;

//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
	const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train, vector<Point2f>& pts_query)
{
	pts_train.clear();
	pts_query.clear();
	pts_train.reserve(matches.size());
	pts_query.reserve(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		const DMatch& match = matches[i];
		pts_query.push_back(kpts_query[match.queryIdx].pt);
		pts_train.push_back(kpts_train[match.trainIdx].pt);
	}

}

static double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, DescriptorMatcher& matcher,
	const Mat& train, const Mat& query, vector<DMatch>& matches)
{

	double t = (double)getTickCount();
	matcher.match(query, train, matches); //Using features2d
	return ((double)getTickCount() - t) / getTickFrequency();
}

int main(int argc, char * argv[])
{
	const char * pPath1 = "D:\\work\\pictures\\dump_shot_input_main_4224x3136_4224x3136.jpg";
	const char * pPath2 = "D:\\work\\pictures\\dump_shot_input_aux_4224x3136_4224x3136.jpg";
	const char * pPath2WarpTo1 = "D:\\work\\pictures\\2WarpTo1.jpg";
	const char * pPath1Sub2 = "D:\\work\\pictures\\1Sub2.jpg";
	Mat imgSrc1 = imread(pPath1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgSrc2 = imread(pPath2, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgSrc1.empty() || imgSrc2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}
	double s = 1.0 / 1;
	Size sz = Size(imgSrc1.rows * s, imgSrc1.cols * s);
	int type = imgSrc1.type();
	Mat img_1(sz, type);
	Mat img_2(sz, type);
	imgSrc1.convertTo(img_1, type, s);
	imgSrc2.convertTo(img_2, type, s);
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 50;

	double t = (double)getTickCount();

	FastFeatureDetector detector(50);
	BriefDescriptorExtractor extractor(32); //this is really 32 x 8 matches since they are binary matches packed into bytes

	vector<KeyPoint> kpts_1, kpts_2;
	detector.detect(img_1, kpts_1);
	detector.detect(img_2, kpts_2);

	t = ((double)getTickCount() - t) / getTickFrequency();

	cout << "found " << kpts_1.size() << " keypoints in " << pPath1 << endl << "fount " << kpts_2.size()
		<< " keypoints in " << pPath2 << endl << "took " << t << " seconds." << endl;

	Mat desc_1, desc_2;

	cout << "computing descriptors..." << endl;

	t = (double)getTickCount();

	extractor.compute(img_1, kpts_1, desc_1);
	extractor.compute(img_2, kpts_2, desc_2);

	t = ((double)getTickCount() - t) / getTickFrequency();

	cout << "done computing descriptors... took " << t << " seconds" << endl;

	//Do matching using features2d
	cout << "matching with BruteForceMatcher<Hamming>" << endl;
	BFMatcher matcher_popcount(NORM_HAMMING);
	vector<DMatch> matches_popcount;
	double pop_time = match(kpts_1, kpts_2, matcher_popcount, desc_1, desc_2, matches_popcount);
	cout << "done BruteForceMatcher<Hamming> matching. took " << pop_time << " seconds" << endl;

	vector<Point2f> mpts_1, mpts_2;
	matches2points(matches_popcount, kpts_1, kpts_2, mpts_1, mpts_2); //Extract a list of the (x,y) location of the matches
	Mat outlier_mask;
	Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);

	Mat outimg;
	drawMatches(img_2, kpts_2, img_1, kpts_1, matches_popcount, outimg, Scalar::all(-1), Scalar::all(-1), outlier_mask);
	namedWindow("matches - popcount - outliers removed", WINDOW_NORMAL);
	imshow("matches - popcount - outliers removed", outimg);

	Mat warped;
	Mat diff;
	warpPerspective(img_2, warped, H, img_1.size());
	namedWindow("warped", WINDOW_NORMAL);
	imshow("warped", warped);
	absdiff(img_1, warped, diff);
	namedWindow("diff", WINDOW_NORMAL);
	imshow("diff", diff);
	imwrite(pPath2WarpTo1, warped);
	imwrite(pPath1Sub2, diff);
	waitKey();

	return 0;
}

