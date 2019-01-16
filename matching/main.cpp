#include <iostream>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <fstream>
#include <stdio.h>
#include <vector>
#include <sstream>
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

#include "timer.h"
#include "kernel.cuh"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void processSurfWithGpu(string objectInputFile, string sceneInputFile, int minHessian, 
std::vector< KeyPoint >&keypoints_scene,std::vector< KeyPoint >&keypoints_object,std::vector< float>&descriptors_scene, std::vector< float>&descriptors_object)
{
	printf("GPU::Processing object: %s and scene: %s ...\n", objectInputFile.c_str(), sceneInputFile.c_str());

	// Load the image from the disk
	Mat img_object = imread( objectInputFile, IMREAD_GRAYSCALE ); // surf works only with grayscale images
	Mat img_scene = imread( sceneInputFile, IMREAD_GRAYSCALE );
	if( !img_object.data || !img_scene.data ) {
		std::cout<< "Error reading images." << std::endl;
		return;
	}

	// Copy the image into GPU memory
	cuda::GpuMat img_object_Gpu( img_object );
	cuda::GpuMat img_scene_Gpu( img_scene );

	// Start the timer - the time moving data between GPU and CPU is added
	GpuTimer timer;
	timer.Start();

	cuda::GpuMat keypoints_scene_Gpu, keypoints_object_Gpu; // keypoints
	cuda::GpuMat descriptors_scene_Gpu, descriptors_object_Gpu; // descriptors (features)

	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	cuda::SURF_CUDA surf( minHessian );
	surf( img_object_Gpu, cuda::GpuMat(), keypoints_object_Gpu, descriptors_object_Gpu );
	surf( img_scene_Gpu, cuda::GpuMat(), keypoints_scene_Gpu, descriptors_scene_Gpu );
	//cout << "FOUND " << keypoints_object_Gpu.cols << " keypoints on object image" << endl;
	//cout << "Found " << keypoints_scene_Gpu.cols << " keypoints on scene image" << endl;

	
	// Downloading results  Gpu -> Cpu
	//vector< KeyPoint > keypoints_scene, keypoints_object;
	//vector< float> descriptors_scene, descriptors_object;
	surf.downloadKeypoints(keypoints_scene_Gpu, keypoints_scene);
	surf.downloadKeypoints(keypoints_object_Gpu, keypoints_object);
	surf.downloadDescriptors(descriptors_scene_Gpu, descriptors_scene);
	surf.downloadDescriptors(descriptors_object_Gpu, descriptors_object);
	
	timer.Stop();
	printf( "Method processWithGpu() ran in: %f msecs, object size: %ux%u, scene size: %ux%u\n",
			timer.Elapsed(), img_object.cols, img_object.rows, img_scene.cols, img_scene.rows );


	//-- Step 8: Release objects from the GPU memory
	surf.releaseMemory();
	img_object_Gpu.release();
	img_scene_Gpu.release();
}
void matchingCPU(int map_num,int img_num,int descriptorDim,std::vector<int>&cpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float** cpu_result,float *cpu_min,int *cpu_index)
{
	///////////////CPU///////////////
	float sum = 0;
	/* float **cpu_result=(float**)malloc(map_num * sizeof(float*));
	for (int i = 0; i<map_num; i++) { cpu_result[i] = (float*)malloc(img_num * sizeof(float*)); } */
	
	cout<<"feature_map"<<feature_map.size()<<endl;
	cout<<"feature_img"<<feature_img.size()<<endl;

	struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
	for (int m = 0; m < map_num; m++)//算描述向量差值
	{
		for (int n = 0; n < img_num; n++)
		{
			if (feature_map[m].laplacian == feature_img[n].laplacian)
			{
				sum = 0;
				for (int k = 0; k < descriptorDim; k++)
				{
					sum += (feature_map[m].descriptor[k] - feature_img[n].descriptor[k])*(feature_map[m].descriptor[k] - feature_img[n].descriptor[k]);
				}
				sum = sqrt(sum);
				cpu_result[m][n] = sum;
				//cout<<m<<setw(5)<<n<<setw(8)<<"Descriptor ok"<<endl;
			}
			else
			{
				cpu_result[m][n] = 10;
			}
			
		}
	}
	/* for(int i=0;i<img_num;i++)
		cout<<cpu_result[0][i]<<endl; */
	float temp_min;	int temp_index; float temp_min2; int temp_index2;
	int a = 0, b = 0; //int *min_index; 
	//static float *minimun;
	//min_index = (int*)malloc(map_num* sizeof(int));
	//minimun = (float*)malloc(map_num* sizeof(float));
	
	for (int a = 0; a < map_num; a++)//算最小值
	{
		b = 0;
		temp_min = cpu_result[a][b]; temp_index = 0;
		for (int b = 1; b < img_num; b++)
		{
			if (temp_min > cpu_result[a][b])
			{
				temp_min = cpu_result[a][b];
				temp_index = b;
			}		
		}
		cpu_min[a] = temp_min;
		cpu_index[a] = temp_index;
	}
/* 	 for(int i=0;i<10;i++)
		cout<<minimun[i]<<endl;  */

	for (int i = 0; i < map_num; i++)//使用窮舉法找出比對到的
	{
		if (cpu_min[i] < 0.09)
		{
			cpu_matcher.push_back(i);
			cpu_matcher.push_back(cpu_index[i]);
		}

	}

	gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);
    
    
	/* for (int i = 0; i < map_num; i++)
		cout<<minimun[i]<<endl; */
    /////////////////free memory /////////////////////////
	//delete cpu_result;
	//delete min_index;
	//delete minimun;
	cout<<"free CPU memory ok"<<endl;
	//turn minimun;
	/////////////////free memory /////////////////////////
}
int main(int argc, char* argv[])
{

    //string fileId = std::to_string(3);
	FILE *fr =fopen("cpu_min","w");
	FILE *fgpu =fopen("gpu_min","w");
	
	int i;
	int keypoints_map_num, keypoints_img_num;
	
	
	Mat out1,out2;
	string mapInputFile = "map.png";
	string imgInputFile = "img.png";
	string outputFile = "matching.png";
	out1 = imread(mapInputFile);
	out2 = imread(imgInputFile);
	
	
    vector< KeyPoint > keypoints_map, keypoints_img;
	vector< float> descriptors_map, descriptors_img;
	
	/////////////////////////////////////////////  surf  /////////////////////////////////////////
    processSurfWithGpu(mapInputFile, imgInputFile,15000,keypoints_map,keypoints_img,descriptors_map,descriptors_img);
    
	
    cout << "FOUND " << keypoints_map.size() << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints_img.size() << " keypoints on second image" << endl;
    cout << "FOUND " << descriptors_map.size() << " descriptors on first image" << endl;
    cout << "FOUND " << descriptors_img.size() << " descriptors on second image" << endl;
	
    /////////////////////////////////////////////put data/////////////////////////////////////////
    keypoints_map_num=keypoints_map.size();
	keypoints_img_num=keypoints_img.size();
	cout << "keypoints_map_num " <<keypoints_map_num<<endl;
	cout << "keypoints_img_num " <<keypoints_img_num<<endl;
	vector<feature>feature_map,feature_img;
    feature temp1,temp2;
    int descriptorDim = 64;
	
    for(int i=0;i<keypoints_map_num;i++)
    {
        memset(&temp1,0,sizeof(feature));//class descriptorDim要手動自己改
		temp1.laplacian=keypoints_map[i].class_id;
        for(int j=0;j<descriptorDim;j++)
			temp1.descriptor[j]=descriptors_map[i*descriptorDim+j];
        feature_map.push_back(temp1);
    }
    cout<<"put data sucessfully 1"<<endl;
    for(int i=0;i<keypoints_img_num;i++)
    {
        memset(&temp2,0,sizeof(feature));
        temp2.laplacian=keypoints_img[i].class_id;
        for(int j=0;j<descriptorDim;j++)
			temp2.descriptor[j]=descriptors_img[i*descriptorDim+j];
        feature_img.push_back(temp2);
    }
	cout<<"put data sucessfully 2"<<endl;
    /////////////////////////////////////////////put data/////////////////////////////////////////
    int couts = 0,thredperblock=NT;
	couts=(keypoints_img_num+thredperblock-1)/thredperblock;
    int *match,match_num=0;
	float *gpu_result;
	float *cpu_min,*gpu_min;
	float *gpu_data_test;
	int *cpu_index,*gpu_index;
	float **cpu_result=(float**)malloc(keypoints_map_num * sizeof(float*));
	for (int i = 0; i<keypoints_map_num; i++) { cpu_result[i] = (float*)malloc(keypoints_img_num * sizeof(float*)); }
	gpu_result=(float*)malloc(keypoints_map_num*keypoints_img_num*sizeof(float));
	cpu_min=(float*)malloc(keypoints_map_num*sizeof(float));
	gpu_min=(float*)malloc(keypoints_map_num * couts*sizeof(float));
	cpu_index=(int*)malloc(keypoints_map_num*sizeof(int));
	gpu_index=(int*)malloc(keypoints_map_num * couts*sizeof(int));
	gpu_data_test=(float*)malloc(keypoints_map_num*keypoints_img_num*sizeof(float));
    
	vector<int>cpu_matcher,gpu_matcher;
	cpu_matcher.clear();
	gpu_matcher.clear();
	//Matching match;
    /*vector<int>match;
	match.clear();*/
	float p=0.001,n=0;
	for(int i=0;i<keypoints_map_num;i++)
	{
		n=0;
		for(int j=0;j<keypoints_img_num;j++)
		{
			gpu_data_test[i*keypoints_img_num+j]=float(j+1)/1000;
		}
		/* for(int j=keypoints_img_num-1;j>=0;j--)
		{
			gpu_data_test[i*keypoints_img_num+j]=p+n;
			n=n+0.001;
		} */
	}
		
	
	matchingCPU(keypoints_map_num,keypoints_img_num,descriptorDim,cpu_matcher,feature_map,feature_img,cpu_result,cpu_min,cpu_index);
	cout<<"matchingCPU sucessfully"<<endl;
    launchKernel(keypoints_map_num,keypoints_img_num,descriptorDim,gpu_matcher,feature_map,feature_img,gpu_result,gpu_min,gpu_index,gpu_data_test);
    //cout<<match_num<<endl;
	int noValue_num=0;
	
	//////////////////////////////////////////check  CPU與GPU算出結果////////////////////////////////////
	
	int error=0,correct=0;
	
	cout<<"cpu_match num"<<cpu_matcher.size()/2<<endl;//兩個成對，前面為第一張feature index , 後面為第二張feature index
	cout<<"gpu_match num"<<gpu_matcher.size()/2<<endl;
    
	/* for(int i=0;i<keypoints_map_num;i++)
	{
		cout<<cpu_min[i]<<endl;
	} */
	/* for(int i=0;i<cpu_matcher.size()/2;i++)
		cout<<"cpu_matcher"<<cpu_matcher[2*i]<<setw(10)<<cpu_matcher[2*i+1]<<endl;
	
	for(int i=0;i<gpu_matcher.size()/2;i++)
		cout<<"gpu_matcher"<<gpu_matcher[2*i]<<setw(10)<<gpu_matcher[2*i+1]<<endl; */
	
	if(cpu_matcher.size()==gpu_matcher.size())
		for(int i=0;i<cpu_matcher.size()/2;i++)
		{
			if(cpu_matcher[2*i]==gpu_matcher[2*i] && cpu_matcher[2*i+1]==gpu_matcher[2*i+1])
				correct++;
			else
			{
				cout<<"cpu_matcher"<<setw(10)<<cpu_matcher[2*i]<<setw(10)<<gpu_matcher[2*i]<<setw(10)<<cpu_matcher[2*i+1]<<setw(10)<<gpu_matcher[2*i+1]<<endl;
				error++;
			}
			
		}
	else
		cout<<"cpu_match num"<<cpu_matcher.size()/2<<setw(10)<<"gpu_match num"<<gpu_matcher.size()/2<<endl;
	
	if(error==0)
		cout<<"matcher sucessfully"<<endl;	
	else
		cout<<"matcher error_num"<<error<<endl;
	/* for(int i=0;i<10;i++)
	{
		cout<<"cpu_map_number"<< setw(5)<<cpu_matcher[2*i]<< setw(15)<< "cpu_matching_num" << setw(5)<< cpu_matcher[2*i+1]<<endl;
		cout<<"gpu_map_number"<< setw(5)<<gpu_matcher[2*i]<< setw(15)<< "gpu_matching_num" << setw(5)<< gpu_matcher[2*i+1]<<endl;
	} */
		
	
	/* float tem_min=0;int tem_index=0;//傳回最後用比CPU比最小
	for(int i=0;i<keypoints_map.size();i++)
	{
		tem_min=gpu_min[couts*i];tem_index = 0;
		for(int b = 1; b < couts; b++)
		{
			if (tem_min > gpu_min[couts*i+b])
			{
				tem_min = gpu_min[couts*i+b];
				//temp_index = b;
			}
		}
		gpu_min[couts*i] = tem_min;
	} */
	
	//////////////////////////////////////////check  CPU與GPU算出結果////////////////////////////////////
	/* for(int i=0;i<keypoints_map_num;i++)
	{
			if(fabs(cpu_min[i]-gpu_min[i*couts])>0.001)
			{
				//correct++;
				cout<<"map_num"<<i<<setw(10)<<"img_num"<<cpu_index[i]<<setw(10)<<"map_num"<<i<<setw(10)<<"img_num"<<gpu_index[i*couts]<<setw(15)<<cpu_min[i]<<setw(10)<<gpu_min[i*couts]<<endl;
				error++;
			}
	}
	if(error==0)
		cout<<"find min sucessfully"<<endl;
	else
		cout<<"find min error_num"<<error<<endl; */
	
	
	for(int j=0;j<keypoints_map_num;j++)
	{
		for(int k=0;k<keypoints_img_num;k++)
		{
			if(fabs(cpu_result[j][k]- gpu_result[j*keypoints_img_num+k])>0.001)
			{
				//cout<<j<<setw(5)<<k<<setw(5)<<"fail"<<endl;
				noValue_num++;
			}
		}
	}

	if(noValue_num==0)
		cout<<"descriptor sucessfully"<<endl;	
	else
		cout<<"descriptor error_num"<<noValue_num<<endl;
		
	for(int i=0;i<keypoints_map_num;i++)
	{
		fprintf(fr,"%f\n",cpu_min[i]);
		fprintf(fgpu,"%f\n",gpu_min[couts*i]);
	}
			
			
	//////////////////////////////////////////畫出特徵點////////////////////////////////////	
    /*for(int i=0;i<match_num;i++)
    {
        cout<<match[2*i]<<setw(5)<<match[2*i+1]<<endl;
    }*/
    /* CvFont Font1=cvFont(1.2,2);
    char num[10];

    for(int i=0;i<match_num;i++)
    {
        Point center1 = Point(keypoints_map[match[2*i]].pt.x,keypoints_map[match[2*i]].pt.y);
        Point center2 = Point(keypoints_img[match[2*i+1]].pt.x,keypoints_img[match[2*i+1]].pt.y);
        Point center1_tex = Point(keypoints_map[match[2*i]].pt.x+10,keypoints_map[match[2*i]].pt.y+10);
        Point center2_tex = Point(keypoints_img[match[2*i+1]].pt.x+10,keypoints_img[match[2*i+1]].pt.y+10);
        //cout<<keypoints1[match[2*i]].octave<<endl;
        circle(out1, center1, keypoints_map[match[2*i]].size/2, Scalar(255,255,0),1); 
        circle(out2, center2, keypoints_img[match[2*i+1]].size/2, Scalar(255,255,0),1);
        sprintf(num,"%d",i);
        putText(out1,num,center1_tex,FONT_HERSHEY_DUPLEX,1,Scalar(255,48,48),1); 
        putText(out2,num,center2_tex,FONT_HERSHEY_DUPLEX,1,Scalar(255,48,48),1); 
        //cvRectangle(out1,cvPoint(keypoints1[match[2*i]].pt.x-r1,keypoints1[match[2*i]].pt.y+r1),cvPoint(keypoints1[match[2*i]].pt.x+r1,keypoints1[match[2*i]].pt.y-r1),Scalar(255,255,0),2,0);
        //cvRectangle(out2,cvPoint(keypoints2[match[2*i+1]].pt.x-r2,keypoints2[match[2*i+1]].pt.y+r2),cvPoint(keypoints2[match[2*i+1]].pt.x+r2,keypoints2[match[2*i+1]].pt.y-r2),Scalar(255,255,0),2,0);
        //cout<<match[2*i]<<setw(5)<<match[2*i+1]<<endl;
    }
    //composite image
    Size img_size = out1.size();
    Mat img_new(img_size.height, img_size.width*2, out1.type() ); Mat part;
    part = img_new(cv::Rect(0, 0, img_size.width, img_size.height));
    out1.copyTo(part);
    part = img_new(cv::Rect(img_size.width, 0, img_size.width, img_size.height));
    out2.copyTo(part);
    //namedWindow(outputFile, 0);
    //imshow(outputFile, img_new);
    imwrite(outputFile,img_new); */
    
    waitKey(0);
	fclose(fr);
	fclose(fgpu);
    return 0;
}