#ifndef _kernel_
#define _kernel_

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
#define NT 64//目前只有64threads
#define GPU_Debug_Create 
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

__global__ void matching(float *d_map, int *l_map, float *d_img, int *l_img, float *result, int img_num,int map_num,int descriptorDim,float *min, int *index,float* gpu_data_test,int *tem_index);
__device__ void findmin1(float *data, int num, float* min, int* index);
__device__ void findmin_global(float *data, int num, float* min, int* index,int* tem_index);
__device__ void findmin1_test(float *data, int num, float* min, int* index);
void launchKernel(int map_num,int img_num,int descriptorDim,std::vector<int>&gpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float*,float* ,int*,float*);
void launchVOKernel(std::vector<keepfeature> matchfeature , std::vector<keepfeature> map_feature,std::vector<Ransac_Pos> &ransac_match,float *,float*,int*,double l_u0,double l_v0,double l_fu,double l_fv);
void launchVOKernel_dub(std::vector<keepfeature> matchfeature , std::vector<keepfeature> map_feature,std::vector<Ransac_Pos> &ransac_match,double *GPU_WL,double *GPU_Rs,int *,double l_u0,double l_v0,double l_fu,double l_fv);
//void processSurfWithGpu(string objectInputFile, string sceneInputFile, int minHessian , std::vector<cv::KeyPoint>&keypoints_scene, std::vector<cv::KeyPoint>&keypoints_object, std::vector<float>&descriptors_scene, std::vector<float>&descriptors_object);
//void matchingCPU(int map_num,int img_num,int descriptorDim,std::vector<int>&cpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float**,float*,int*);

#endif
