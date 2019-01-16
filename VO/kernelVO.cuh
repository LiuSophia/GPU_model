#ifndef _VOkernel_
#define _VOkernel_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>

#include <stdio.h>
#include <vector>
#include <iomanip>
#include <sys/time.h>
#include <string>
#include <sstream>
#include "opencv2/opencv_modules.hpp"
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

#include "MapManagement.h"
#include "RANSAC.h"
#define NT 128//目前只有64threads
#define GPU_Debug_Create 
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int launchVOKernel(std::vector<Keep_Feature> matchfeature , std::vector<Keep_Feature> map_feature,float *CameraPose,int *,double l_u0,double l_v0,double l_fu,double l_fv);
void launchVOKernel_dub(std::vector<Keep_Feature> matchfeature , std::vector<Keep_Feature> map_feature,double *CameraPose,double l_u0,double l_v0,double l_fu,double l_fv);

#endif