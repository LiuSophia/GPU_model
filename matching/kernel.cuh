#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <sys/time.h>
#include <string>

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

#define NT 16//目前只有64threads
#define GPU_Debug_Create 
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class feature {
public:
	int laplacian;
	float descriptor[64] = {0};
	int ix;
	int iy;
	int num;

};

class Matching{
public:
    int match_num;
    int matcher[];
};

__global__ void matching(float *d_map, int *l_map, float *d_img, int *l_img, float *result, int *match, int img_num, int map_num, float *min, int *index);
__device__ void findmin1(float *data, int num, float* min, int* index);
__device__ void findmin2(float *data, float* min, int* index, int num);
void launchKernel(int map_num,int img_num,int descriptorDim,std::vector<int>&gpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float*,float* ,int*,float*);
void processSurfWithGpu(string objectInputFile, string sceneInputFile, int minHessian , std::vector<cv::KeyPoint>&keypoints_scene, std::vector<cv::KeyPoint>&keypoints_object, std::vector<float>&descriptors_scene, std::vector<float>&descriptors_object);
void matchingCPU(int map_num,int img_num,int descriptorDim,std::vector<int>&cpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float**,float*,int*);