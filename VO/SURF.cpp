#include "SURF.h"

void OpevCvSURF::processSurfWithGpu(const Mat l_image, int minHessian,std::vector< KeyPoint >&keypoints,std::vector< float>&descriptors)
{
	//printf("GPU::Processing object: %s and scene: %s ...\n", objectInputFile.c_str(), sceneInputFile.c_str());

	// Copy the image into GPU memory
	cuda::GpuMat l_image_GPU( l_image );

	// Start the timer - the time moving data between GPU and CPU is added
	GpuTimer timer;
	timer.Start();

	cuda::GpuMat keypoints_Gpu; // keypoints
	cuda::GpuMat descriptors_Gpu; // descriptors (features)
	int _nOctaves = 4;
	int _nOctaveLayers = 2;
	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	cuda::SURF_CUDA surf( minHessian,_nOctaves ,_nOctaveLayers);
	surf( l_image_GPU, cuda::GpuMat(), keypoints_Gpu, descriptors_Gpu );

	//cout << "FOUND " << keypoints_Gpu.cols << " keypoints on image" << endl;

	// Downloading results  Gpu -> Cpu
	surf.downloadKeypoints(keypoints_Gpu, keypoints);
	surf.downloadDescriptors(descriptors_Gpu, descriptors);
	
	timer.Stop();
	printf( "Method processWithGpu() ran in: %f msecs, l_image size: %ux%u\n",timer.Elapsed(), l_image.cols, l_image.rows);

	//-- Step 8: Release objects from the GPU memory
	surf.releaseMemory();
	l_image_GPU.release();
}


void OpevCvSURF::processSurfWithCpu(const Mat l_image, int minHessian,std::vector< KeyPoint >&keypoints,Mat &descriptors)
{
	//printf("CPU::Processing object: %s and scene: %s ...\n", objectInputFile.c_str(), sceneInputFile.c_str());

	// Load the image from the disk
	//Mat img_object = imread( objectInputFile, IMREAD_GRAYSCALE ); // surf works only with grayscale images
	//Mat img_scene = imread( sceneInputFile, IMREAD_GRAYSCALE );

	if( !l_image.data) {
		std::cout<< "Error processSurfWithCpu with images." << std::endl;
		return;
	}

	// Start the timer
	GpuTimer timer;
	timer.Start();

	int nOctaves = 4;
	int nOctaveLayers = 2;
	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	Ptr<SURF> surf = SURF::create( minHessian ,nOctaves,nOctaveLayers);
	surf->detectAndCompute( l_image, noArray(), keypoints, descriptors );
	
	timer.Stop();
	printf( "Method processWithCpu() ran in: %f msecs, object size: %ux%u\n",
			timer.Elapsed(), l_image.cols, l_image.rows);

}