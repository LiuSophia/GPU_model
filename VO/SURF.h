#ifndef _SURF_h_
#define _SURF_h_


#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "timer.h"
#include <sstream>
#include <string>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class OpevCvSURF
{
public:
	void processSurfWithGpu(const Mat l_image, int minHessian,std::vector< KeyPoint >&keypoints,std::vector< float>&descriptors);
	void processSurfWithCpu(const Mat l_image, int minHessian,std::vector< KeyPoint >&keypoints,Mat &descriptors);

};

#endif