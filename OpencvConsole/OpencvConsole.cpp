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
#pragma comment(lib, "opencv_photo2413d.lib")

//#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"

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



#define PICS_DIR_NLMFNR "D:\\work\\pictures\\denoise_iso_9600_x2\\"

int cmp(const void * pData1, const void * pData2)
{
	return *(const uchar*)pData1 - *(const uchar*)pData2;
}

uchar GetMedian(uchar * pData, int nLen)
{
	uchar nRet = 0;
	qsort(pData, nLen, 1, cmp);
	if (nLen % 2 == 0)
	{
		int value = (pData[nLen / 2 - 1] + pData[nLen / 2]) / 2;
		nRet = value;
	}
	else
	{
		nRet = pData[nLen / 2];
	}
	return nRet;
}

int NLMFNR()
{
	const int imgs_count = 3;
	vector<Mat> original(imgs_count);
	original[0] = imread(PICS_DIR_NLMFNR"letv_9600(1)_[]_2015-12-08_12-56-26.jpg", CV_LOAD_IMAGE_COLOR);
	original[1] = imread(PICS_DIR_NLMFNR"letv_9600(2)_[]_2015-12-08_12-56-29.jpg", CV_LOAD_IMAGE_COLOR);
	original[2] = imread(PICS_DIR_NLMFNR"letv_9600(3)_[]_2015-12-08_12-56-31.jpg", CV_LOAD_IMAGE_COLOR);
	IplImage * pIplImage0 = &IplImage(original[0]);
	IplImage * pIplImage1 = &IplImage(original[1]);
	IplImage * pIplImage2 = &IplImage(original[2]);

	Mat result(cvGetSize(pIplImage0), CV_8UC3);
	IplImage * pIplImageResult = &IplImage(result);
	for (int j = 0; j < pIplImage0->height; j++)
	{
		uchar * pLineImage0 = (uchar*)pIplImage0->imageData + pIplImage0->widthStep * j;
		uchar * pLineImage1 = (uchar*)pIplImage1->imageData + pIplImage1->widthStep * j;
		uchar * pLineImage2 = (uchar*)pIplImage2->imageData + pIplImage2->widthStep * j;

		uchar * pLineImageResult = (uchar*)pIplImageResult->imageData + pIplImageResult->widthStep * j;
		for (int i = 0; i < pIplImage0->width; i++)
		{
#if 0
			uchar bs[3] = { pLineImage0[i * 3], pLineImage1[i * 3], pLineImage2[i * 3] };
			uchar gs[3] = { pLineImage0[i * 3+1], pLineImage1[i * 3+1], pLineImage2[i * 3+1] };
			uchar rs[3] = { pLineImage0[i * 3+2], pLineImage1[i * 3+2], pLineImage2[i * 3+2] };
			pLineImageResult[i * 3] = GetMedian(bs, sizeof(bs));
			pLineImageResult[i * 3 + 1] = GetMedian(gs, sizeof(gs));
			pLineImageResult[i * 3 + 2] = GetMedian(rs, sizeof(rs));
#else
			int b = (pLineImage0[i*3] + pLineImage1[i*3] + pLineImage2[i*3]) / 3;
			int g = (pLineImage0[i*3+1] + pLineImage1[i*3+1] + pLineImage2[i*3+1]) / 3;
			int r = (pLineImage0[i*3+2] + pLineImage1[i*3+2] + pLineImage2[i*3+2]) / 3;
			pLineImageResult[i*3] = b;
			pLineImageResult[i*3+1] = g;
			pLineImageResult[i*3+2] = r;
#endif
		}
	}
	//fastNlMeansDenoisingColoredMulti(original, result, imgs_count / 2, imgs_count, 10, 15);
	imwrite(PICS_DIR_NLMFNR"result.png", result);
	return 0;
}

static void RemoveVectorMoreThan(vector<KeyPoint> & kpts_1, int up)
{
	if (kpts_1.size() > up)
	{
		auto it = kpts_1.begin();
		for (int i = 0; i < up; i++)
		{
			it++;
		}
		kpts_1.erase(it, kpts_1.end());
	}
}

//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_020903_161\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_060641_135\\"

//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_061948_250\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_062015_417\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_062327_704\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_062404_093\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_213535_277\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_213548_761\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_214239_406\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_215245_839\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700102_215254_136\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_011258_661\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_015214_170\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_020903_161\\"
//#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_021106_970\\"/*夜景，彩图清晰，黑白图模糊*/
#define PICS_DIR "D:\\work\\pictures\\AF sync\\19700103_021115_789\\"

int main(int argc, char * argv[])
{
	extern int dwt_main();
	//return dwt_main();
	//return NLMFNR();
	const char * pPath1 = PICS_DIR"dump_shot_input_main_4224x3136_4224x3136.jpg";
	const char * pPath2 = PICS_DIR"dump_shot_input_aux_4224x3136_4224x3136.jpg";
	const char * pPath2WarpTo1 = PICS_DIR"2WarpTo1.jpg";
	const char * pPath1Sub2 = PICS_DIR"1Sub2warped.jpg";
	Mat imgSrc1 = imread(pPath1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgSrc2 = imread(pPath2, CV_LOAD_IMAGE_GRAYSCALE);
	if (imgSrc1.empty() || imgSrc2.empty())
	{
		printf("Can't read one of the images\n");
		return -1;
	}
	double s = 1.0 / 1;
	Size sz = Size(imgSrc1.cols * s, imgSrc1.rows * s);
	int type = imgSrc1.type();
	Mat img_1(sz, type);
	Mat img_2(sz, type);
	resize(imgSrc1, img_1, sz, s, s);
	resize(imgSrc2, img_2, sz, s, s);
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 50;

	double t = (double)getTickCount();

	OrbFeatureDetector detector(6000);
	BriefDescriptorExtractor extractor(32); //this is really 32 x 8 matches since they are binary matches packed into bytes

	vector<KeyPoint> kpts_1, kpts_2;
	detector.detect(img_1, kpts_1);
	detector.detect(img_2, kpts_2);

	//RemoveVectorMoreThan(kpts_1, 1000);
	//RemoveVectorMoreThan(kpts_2, 1000);

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
#if 1
	Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);
	cout << H << endl;
	FileStorage fs(PICS_DIR"mat.txt", FileStorage::WRITE);
	fs << "homography" << H;
	fs.release();
#else
	Mat H;
	H.create(3, 3, CV_32FC1);
	FileStorage fs("D:\\work\\pictures\\AF sync\\19700102_062327_704\\mat.txt", FileStorage::READ);
	fs["homography"] >> H;
#endif
	Mat outimg;
	drawMatches(img_2, kpts_2, img_1, kpts_1, matches_popcount, outimg, Scalar::all(-1), Scalar::all(-1), outlier_mask);
	namedWindow("matches - popcount - outliers removed", WINDOW_NORMAL);
	imshow("matches - popcount - outliers removed", outimg);

	Mat warped;
	Mat diff;
	warpPerspective(img_2, warped, H, img_1.size(), 1, 0, Scalar(255.));
	
	Scalar_<int> contour(warped.cols, warped.rows, 0, 0);
	Rect clip;
	for (int j = 0; j < warped.rows; j++)
	{
		for (int i = 0; i < warped.cols; i++)
		{
			uchar v = warped.at<uchar>(j, i);
			if (v == 0)
			{
				if (i < contour[0])
				{
					contour[0] = i;
				}

				if (j < contour[1])
				{
					contour[1] = j;
				}

				if (i > contour[2])
				{
					contour[2] = i;
				}

				if (j > contour[3])
				{
					contour[3] = j;
				}
			}
		}
	}
	clip.x = contour[0];
	clip.y = contour[1];
	clip.width = contour[2] - contour[0];
	clip.height = contour[3] - contour[1];

	namedWindow("warped", WINDOW_NORMAL);
	imshow("warped", warped);
	absdiff(img_1, warped, diff);
	namedWindow("diff", WINDOW_NORMAL);
	imshow("diff", diff);
	imwrite(pPath2WarpTo1, warped);
	imwrite(pPath1Sub2, diff);
	imwrite(PICS_DIR"1.jpg", imgSrc1);
	imwrite(PICS_DIR"2.jpg", imgSrc2);

	waitKey();

	return 0;
}

