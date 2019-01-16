
#include "kernel.cuh"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace std;



__device__ void findmin1_test(float *data, int num, float* min, int* index) {
	
	int tid = threadIdx.x;
	int blockid = blockIdx.x;
	int couts =0;//一個block需要做的次數(從所有data)
	int min_couts=0;//一個block需要做的次數(從min)

	couts=(num+blockDim.x-1)/blockDim.x;

	
	__shared__ int k;
	__shared__ int m;
	__shared__ int tem_index[NT/2];
	__shared__ float tem_descriptor[NT/2] ;
	k = 0;
	
		while (k<couts)//判斷k<couts
		{
			if (num - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;
				m = 0;
				while (i != 0)
				{
					if (tid < i) {
						if (m == 0)//第一次先放進share memory
						{
							if (data[blockid * num + k*blockDim.x + tid] > data[blockid * num + k*blockDim.x + tid + i])
							{
								tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid + i];
								tem_index[tid] = k*blockDim.x + tid + i;
							}
							else
							{
								tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid];
								tem_index[tid] = k*blockDim.x + tid;
							}
							__syncthreads();
						}
						else//第二次從share memory 拿值比
						{
							if (tem_descriptor[tid+ i ] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + i];
								tem_index[tid]= tem_index[tid + i];
							}
						}
					}
					__syncthreads();
					i /= 2;
					m++;
				}
			}
			else//處理多餘
			{
				int remain = num - blockDim.x*k;
				int tem_j = remain;
				int j = remain / 2;
				m = 0;
				while (j != 0)
				{
					if (tid < j) 
					{
						if (m == 0)//第一次先放進share memory
						{
							if (data[blockid * num + k*blockDim.x + tid] > data[blockid * num + k*blockDim.x + tid + j])
							{
								tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid + j];
								tem_index[tid] = k*blockDim.x + tid + j;
							}
							else
							{
								tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid];
								tem_index[tid] = k*blockDim.x + tid;
							}
							if (tid + j * 2 < tem_j) {
									if (data[blockid * num + k*blockDim.x + tid + j * 2] < tem_descriptor[tid])//多餘的要跟已經放shared的比
									{
										tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid + j * 2];
										tem_index[tid] = k*blockDim.x + remain - 1;
									}
								}
							__syncthreads();
						}
						else//第二次從share memory 拿值比
						{
							if (tem_descriptor[tid+j] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + j];
								tem_index[tid] = tem_index[tid + j];
							}
							if (tid + j * 2 < tem_j) {
								if (tem_descriptor[tid + j * 2] < tem_descriptor[tid])
								{
									tem_descriptor[tid] = tem_descriptor[tid + j * 2];
									tem_index[tid] = tem_index[tid + j * 2];
								}
							}
						}
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid ];
						tem_index[tid] = k*blockDim.x + tid;
					}
				}
			}
			if(tid==0)
			{
				min[blockid * couts + k] = tem_descriptor[0];
				index[blockid * couts + k] = tem_index[0];
			}
			__syncthreads();
			k++;
		}

		min_couts=(couts+blockDim.x-1)/blockDim.x;
		k=0;
		
		while(k<min_couts)
		{
			if(couts - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;
				m = 0;
				while (i != 0)
				{
					if (tid < i) {
						if(m==0)
						{
							if(min[blockid*couts +k*blockDim.x+tid]<min[blockid*couts +k*blockDim.x+tid+i])
							{
								tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid];
								tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
							}
							else
							{
								tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid+i];
								tem_index[tid] = index[blockid*couts + k*blockDim.x + tid+i];
							}
						__syncthreads();
						}
						else
						{
							if (tem_descriptor[tid+ i ] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + i];
								tem_index[tid]= tem_index[tid + i];
							}
						}
						
					}
					__syncthreads();
					i /= 2;
					m++;
				}
			}
			else
			{
				int remain = couts - blockDim.x*k;//7
				int tem_j = remain;
				int j = remain / 2;//3
				m = 0;
				while (j != 0)
				{
					if (tid < j) 
					{
						if(m==0)
						{
							if(min[blockid*couts +k*blockDim.x+tid]<min[blockid*couts +k*blockDim.x+tid+j])
							{
								tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid];
								tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
							}
							else
							{
								tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid+j];
								tem_index[tid] = index[blockid*couts + k*blockDim.x + tid+j];
							}
							if (tid + j * 2 < tem_j) {
								if (min[blockid*couts +k*blockDim.x+tid + j * 2] < tem_descriptor[tid])
								{
									tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid + j * 2];
									tem_index[tid] = index[blockid*couts + k*blockDim.x + tid + j * 2];
								}
							}
						__syncthreads();
						}
						else
						{
							if (tem_descriptor[tid+j] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + j];
								tem_index[tid] = tem_index[tid + j];
							}
							if (tid + j * 2 < tem_j) {
								if (tem_descriptor[tid + j * 2] < tem_descriptor[tid])
								{
									tem_descriptor[tid] = tem_descriptor[tid + j * 2];
									tem_index[tid] = tem_index[tid + j * 2];
								}
							}
						}
							
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						tem_descriptor[tid] = min[blockid * couts + k*blockDim.x + tid ];
						tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
					}
				}
				
			}
			
			if(tid==0)
			{
				min[blockid * couts + k] = tem_descriptor[0];
				index[blockid * couts + k] = tem_index[0];
			}
			k++;

		}
		/* if (tid < couts)
		{
			int i = couts / 2;
			int tem_i = couts;
			while (i != 0)
			{
				if (tid < i)
				{
					if (min[blockid*couts + tid] > min[blockid*couts + tid + i])
					{
						min[blockid*couts + tid] = min[blockid*couts + tid + i];
						index[blockid*couts + tid] = index[blockid*couts + tid + i];
					}
					if (tid + i * 2 < tem_i) {
						if (min[blockid*couts + tid] >min[blockid*couts + tid + 2 * i])
						{
							min[blockid*couts + tid] = min[blockid*couts + tid + 2 * i];
							index[blockid*couts + tid] = index[blockid*couts + tid + 2 * i];
						}
					}//如果多餘的是奇數
				}
				__syncthreads();
				i /= 2;
				tem_i /= 2;
			}
		} */
}

__device__ void findmin_global(float *data, int num, float* min, int* index,int* tem_index)
{
	int tid = threadIdx.x;
	int blockid = blockIdx.x;
	int couts =0;//一個block需要做的次數(從所有data)
	int min_couts=0;//一個block需要做的次數(從min)
	couts=(num+blockDim.x-1)/blockDim.x;
	int k=0,m=0;
	//__shared__ int tem_index[NT/2];//global
	
		while (k<couts)//判斷k<couts
		{
			if (num - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;
				m=0;
				while (i != 0)
				{
					if (tid < i) {
						if (data[blockid * num + k*blockDim.x + tid] > data[blockid * num + k*blockDim.x + tid + i])
						{
							data[blockid * num + k*blockDim.x + tid] = data[blockid * num + k*blockDim.x + tid + i];
							if(m==0)	tem_index[blockid*NT/2+tid] = k*blockDim.x + tid + i;
							else	tem_index[blockid*NT/2+tid] = tem_index[blockid*NT/2+tid+i];
						}
						else
						{
							if(m==0)	tem_index[blockid*NT/2+tid] = k*blockDim.x + tid;
						}	
					}
					__syncthreads();
					i /= 2;
					m++;
				}
			}
			else//處理多餘
			{
				int remain = num - blockDim.x*k;
				int tem_j = remain;
				int j = remain / 2;
				m=0;
				while (j != 0)
				{
					if (tid < j) 
					{
						
						if (data[blockid * num + k*blockDim.x + tid] > data[blockid * num + k*blockDim.x + tid + j])
						{
							data[blockid * num + k*blockDim.x + tid] = data[blockid * num + k*blockDim.x + tid + j];
							/* if(m==0)	tem_index[tid] = k*blockDim.x + tid + j;
							else	tem_index[tid] = tem_index[tid+j]; */
							if(m==0)	tem_index[blockid*NT/2+tid] = k*blockDim.x + tid + j;
							else	tem_index[blockid*NT/2+tid] = tem_index[blockid*NT/2+tid+j];
						}
						else
						{
							if(m==0)	tem_index[blockid*NT/2+tid] = k*blockDim.x + tid;
						}
						if (tid + j * 2 < tem_j) {
							if (data[blockid * num + k*blockDim.x + tid + j * 2] < data[blockid * num + k*blockDim.x + tid])
							{
								data[blockid * num + k*blockDim.x + tid] = data[blockid * num + k*blockDim.x + tid + j * 2];
								/* if(m==0)	tem_index[tid] = k*blockDim.x + remain - 1;
								else	tem_index[tid] = tem_index[tid+j*2]; */
								if(m==0)	tem_index[blockid*NT/2+tid] = k*blockDim.x + remain - 1;
								else	tem_index[blockid*NT/2+tid] = tem_index[blockid*NT/2+tid+j*2];
							}
						}
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						data[blockid * num + k*blockDim.x + tid] = data[blockid * num + k*blockDim.x + tid ];
						tem_index[blockid*NT/2+tid] = k*blockDim.x + tid;
					}
				}
			}
			if(tid==0)
			{
				min[blockid * couts + k] = data[blockid * num + k*blockDim.x];
				index[blockid * couts + k] = tem_index[blockid*NT/2];
			}
			
			k++;
		}
		
		min_couts=(couts+blockDim.x-1)/blockDim.x;//3
		k=0;
		
		while(k<min_couts)
		{
			if(couts - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;

				while (i != 0)
				{
					if (tid < i) {
							if (min[blockid*couts +k*blockDim.x+tid+ i ] < min[blockid*couts +k*blockDim.x+tid])
							{
								min[blockid*couts +k*blockDim.x+tid] = min[blockid*couts +k*blockDim.x+tid+ i ];
								index[blockid*couts +k*blockDim.x+tid]= index[blockid*couts +k*blockDim.x+tid + i];
							}
					}
					__syncthreads();
					i /= 2;
					//m++;
				}
			}
			else
			{
				int remain = couts - blockDim.x*k;
				int tem_j = remain;
				int j = remain / 2;
				
				while (j != 0)
				{
					if (tid < j) 
					{
							if (min[blockid*couts +k*blockDim.x+tid+j] < min[blockid*couts +k*blockDim.x+tid])
							{
								min[blockid*couts +k*blockDim.x+tid] = min[blockid*couts +k*blockDim.x+tid + j];
								index[blockid*couts +k*blockDim.x+tid] = index[blockid*couts +k*blockDim.x+tid + j];
							}
							if (tid + j * 2 < tem_j) {
								if (min[blockid*couts +k*blockDim.x+tid + j * 2] < min[blockid*couts +k*blockDim.x+tid])
								{
									min[blockid*couts +k*blockDim.x+tid] = min[blockid*couts +k*blockDim.x+tid + j * 2];
									index[blockid*couts +k*blockDim.x+tid] = index[blockid*couts +k*blockDim.x+tid + j * 2];
								}
							}
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					//m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						min[blockid*couts +k*blockDim.x+tid] = min[blockid * couts + k*blockDim.x + tid ];
						index[blockid*couts +k*blockDim.x+tid] = index[blockid*couts + k*blockDim.x + tid];
					}
				}
				
			}
			
			if(tid==0)
			{
				min[blockid * couts + k] = min[blockid*couts +k*blockDim.x];
				index[blockid * couts + k] = index[blockid*couts +k*blockDim.x];
			}
			k++;

		}
		
}

__device__ void findmin1(float *data, int num, float* min, int* index) {
	
	int tid = threadIdx.x;
	int blockid = blockIdx.x;
	int couts =0;//一個block需要做的次數(從所有data)
	int min_couts=0;//一個block需要做的次數(從min)
	couts=(num+blockDim.x-1)/blockDim.x;

	
	__shared__ int k;//block做了第幾次
	//__shared__ int m;
	__shared__ int tem_index[NT];
	__shared__ float tem_descriptor[NT] ;
	
	k = 0; //m = 0;
	
		while (k<couts)//判斷k<couts
		{
			if (num - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;
				//m = 0;
				//if (m == 0)//第一次先放進share memory
				//{
					tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid];
					tem_index[tid] = k*blockDim.x + tid;
					__syncthreads();
				//}
				
				while (i != 0)
				{
					if (tid < i) {
							if (tem_descriptor[tid+ i ] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + i];
								tem_index[tid]= tem_index[tid + i];
							}
					}
					__syncthreads();
					i /= 2;
					//m++;
				}
			}
			else//處理多餘
			{
				int remain = num - blockDim.x*k;
				int tem_j = remain;
				int j = remain / 2;
				//m = 0;
				if (/*m == 0 && */tid<remain)//第一次先放進share memory
				{
					tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid];
					tem_index[tid] = k*blockDim.x + tid;
					__syncthreads();
				}
				
				while (j != 0)
				{
					if (tid < j) 
					{
							if (tem_descriptor[tid+j] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + j];
								tem_index[tid] = tem_index[tid + j];
							}
							if (tid + j * 2 < tem_j) {
								if (tem_descriptor[tid + j * 2] < tem_descriptor[tid])
								{
									tem_descriptor[tid] = tem_descriptor[tid + j * 2];
									tem_index[tid] = tem_index[tid + j * 2];
								}
							}
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					//m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						tem_descriptor[tid] = data[blockid * num + k*blockDim.x + tid ];
						tem_index[tid] = k*blockDim.x + tid;
					}
				}
			}
			
			if(tid==0)
			{
				min[blockid * couts + k] = tem_descriptor[0];
				index[blockid * couts + k] = tem_index[0];
			}
			
			k++;
		}

		min_couts=(couts+blockDim.x-1)/blockDim.x;//3
		k=0;
		
		while(k<min_couts)
		{
			if(couts - blockDim.x*k >= blockDim.x)
			{
				int i = blockDim.x / 2;
				//m = 0;
				//if(m==0)
				//{
					tem_descriptor[tid] = min[blockid*couts +k*blockDim.x+tid];
					tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
					__syncthreads();
				//}
				while (i != 0)
				{
					if (tid < i) {
							if (tem_descriptor[tid+ i ] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + i];
								tem_index[tid]= tem_index[tid + i];
							}
					}
					__syncthreads();
					i /= 2;
					//m++;
				}
			}
			else
			{
				int remain = couts - blockDim.x*k;
				int tem_j = remain;
				int j = remain / 2;
				//m = 0;
				if (/*m == 0 &&*/ tid<remain)//第一次先放進share memory
				{
					tem_descriptor[tid] = min[blockid*couts + k*blockDim.x + tid];
					tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
					__syncthreads();
				}
				
				while (j != 0)
				{
					if (tid < j) 
					{
							if (tem_descriptor[tid+j] < tem_descriptor[tid])
							{
								tem_descriptor[tid] = tem_descriptor[tid + j];
								tem_index[tid] = tem_index[tid + j];
							}
							if (tid + j * 2 < tem_j) {
								if (tem_descriptor[tid + j * 2] < tem_descriptor[tid])
								{
									tem_descriptor[tid] = tem_descriptor[tid + j * 2];
									tem_index[tid] = tem_index[tid + j * 2];
								}
							}
					}
					__syncthreads();
					tem_j /= 2;
					j /= 2;
					//m++;
				}
				if(remain==1)//如果剩一個
				{
					if (tid < remain)
					{
						tem_descriptor[tid] = min[blockid * couts + k*blockDim.x + tid ];
						tem_index[tid] = index[blockid*couts + k*blockDim.x + tid];
					}
				}
				
			}
			
			if(tid==0)
			{
				min[blockid * couts + k] = tem_descriptor[0];
				index[blockid * couts + k] = tem_index[0];
			}
			k++;

		}
		

}

__global__ void matching(float *d_map, int *l_map, float *d_img, int *l_img, float *result, int img_num,int map_num,int descriptorDim,float *min, int *index,float* gpu_data_test,int *tem_index)
{
	int tid = threadIdx.x;
	int temp_tid = 0;
	float SAD = 0.0;
	__shared__ float sum[NT] ;
	
		for (int j = 0; j < img_num; j++)
		{
			temp_tid=tid;
			SAD = 0.0;
			if (l_map[blockIdx.x] == l_img[j])
			{
				 for (int i = 0; i < descriptorDim/NT; i++)
				{ 
				//SAD = (d_map[blockIdx.x *descriptorDim+ tid] - d_img[j*descriptorDim + tid])*(d_map[blockIdx.x *descriptorDim+ tid] - d_img[j*descriptorDim + tid]);
					 SAD += (d_map[blockIdx.x *descriptorDim+ temp_tid] - d_img[j*descriptorDim + temp_tid])*(d_map[blockIdx.x *descriptorDim+ temp_tid] - d_img[j*descriptorDim + temp_tid]);
					temp_tid += blockDim.x;
				} 
				sum[tid] = SAD;
				__syncthreads();

				int i = blockDim.x / 2;
				while (i != 0)
				{
					if (tid < i) {
						sum[tid] += sum[tid + i];
					}
					__syncthreads();
					i /= 2;
				}
				
				if(tid==0)
					sum[0] = sqrt(sum[0]);
				__syncthreads();
				
				result[img_num*blockIdx.x + j] = sum[0];
			}
			else
			{
				result[img_num*blockIdx.x + j] = 10.0;
			}

		}

	//findmin1(result, img_num, min, index);//找最小 shared memory=threads數
	findmin_global(gpu_data_test, img_num, min, index,tem_index);//找最小 global memory
	//findmin1_test(result, img_num, min, index); //找最小 shared memory threads數一半
	
		

}


void launchKernel(int map_num,int img_num,int descriptorDim,std::vector<int>&gpu_matcher,std::vector<feature>&feature_map,std::vector<feature>&feature_img,float* gpu_result,float* gpu_min,int* gpu_index,float* gpu_data_test)
{
		
	int blocknum = map_num;
	int thredperblock = NT;
	int couts = 0;
		
	couts=(img_num+thredperblock-1)/thredperblock;
    
    
    float *d_map, *d_img ;
	int  *l_map, *l_img;
	/////////////////// allocate memory on the cpu side//////////////////////
	d_map = (float*)malloc(map_num *descriptorDim* sizeof(float));
	l_map = (int*)malloc(map_num * sizeof(int));
	d_img = (float*)malloc(img_num *descriptorDim* sizeof(float));
	l_img = (int*)malloc(img_num * sizeof(int));
    ////////////////// put data in the host memory////////////////////////
    for (int i = 0; i < map_num; i++)
    {
        l_map[i]=feature_map[i].laplacian;
        for(int j=0;j<descriptorDim;j++)
		    d_map[i*descriptorDim+j]=feature_map[i].descriptor[j];
    }
        
	for (int i = 0; i < img_num; i++)
    {
        l_img[i]=feature_img[i].laplacian;
        for(int j=0;j<descriptorDim;j++)
		    d_img[i*descriptorDim+j]=feature_img[i].descriptor[j];
    }	
	////////////////// put data in the host memory////////////////////////
    
	float *dev_d_map, *dev_d_img;
	int *dev_l_map, *dev_l_img;
	float *dev_result;
	float *dev_min;
	int  *dev_index;
	float *dev_gpu_data_test;
	int *global_tem_index,*dev_tem_index;
	global_tem_index=(int*)malloc(map_num*NT/2* sizeof(int));
	////////////////// allocate the memory on the GPU ///////////////////////
	
	cudaMalloc((void **)&dev_d_map, map_num *descriptorDim* sizeof(float));
	cudaMalloc((void **)&dev_l_map, map_num * sizeof(int));
	cudaMalloc((void **)&dev_d_img, img_num * descriptorDim*sizeof(float));
	cudaMalloc((void **)&dev_l_img, img_num * sizeof(int));
	cudaMalloc((void **)&dev_result, img_num*map_num * sizeof(float));
	//cudaMalloc((void **)&dev_match, 2 * map_num * sizeof(int));
	cudaMalloc((void **)&dev_min, couts*map_num * sizeof(float));
	cudaMalloc((void **)&dev_index, couts * map_num * sizeof(int));
	cudaMalloc((void **)&dev_gpu_data_test,map_num*img_num* sizeof(float));
	cudaMalloc((void **)&dev_tem_index,map_num*NT/2* sizeof(int));
	////////////////// copy the arrays 's' and 'p' to the GPU/////////////////////////
	cudaMemcpy(dev_d_map, d_map, map_num * descriptorDim*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_l_map, l_map, map_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_d_img, d_img, img_num *descriptorDim* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_l_img, l_img, img_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_result, gpu_result, img_num*map_num * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_match, match, 2 * map_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_min, gpu_min, couts*map_num * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_index, gpu_index, couts*map_num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gpu_data_test,gpu_data_test,map_num*img_num* sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_tem_index,global_tem_index,map_num*NT/2* sizeof(int),cudaMemcpyHostToDevice);
	// Get start time event
	cudaEvent_t start_kernel, stop_kernel;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventRecord(start_kernel, 0);

	matching << < blocknum, thredperblock >> > (dev_d_map, dev_l_map, dev_d_img, dev_l_img, dev_result, img_num, map_num,descriptorDim,dev_min, dev_index,dev_gpu_data_test,dev_tem_index);

	
	// Get stop time event
	cudaEventRecord(stop_kernel, 0);
	cudaEventSynchronize(stop_kernel);
	// Compute execution time
	float kernelTime;
	cudaEventElapsedTime(&kernelTime, start_kernel, stop_kernel);
	printf("GPU time: %13f msec\n", kernelTime);
	cudaEventDestroy(start_kernel);
	cudaEventDestroy(stop_kernel);
	

	cudaError_t cuda_err = cudaGetLastError();
	if (cudaSuccess != cuda_err) {
		cout << "before kernel call: error = %s\n" << cudaGetErrorString(cuda_err) << endl;
		system("pause");
		exit(1);
	}

	cudaEvent_t start_tra, stop_tra;
	cudaEventCreate(&start_tra);
	cudaEventCreate(&stop_tra);
	cudaEventRecord(start_tra, 0);
	
	cudaMemcpy(gpu_result, dev_result, img_num*map_num * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_min, dev_min, couts*map_num * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_index, dev_index, couts*map_num *sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop_tra, 0);
	cudaEventSynchronize(stop_tra);

	// Compute execution time
	float transferTime;
	cudaEventElapsedTime(&transferTime, start_tra, stop_tra);
	printf("GPU transfer time: %13f msec\n", transferTime);
	cudaEventDestroy(start_tra);
	cudaEventDestroy(stop_tra);
	
	
		
	int min_couts=0;
	float tem_min=0;
	int tem_index=0;
	
	min_couts=(couts+thredperblock-1)/thredperblock;
	/* for(int j=0;j<map_num;j++)
		for(int i=0;i<min_couts;i++)
			cout<<gpu_min[j*couts+i]<<endl; */
	/* for(int j=0;j<12;j++)
			cout<<gpu_index[j]<<endl;
		cout<<endl; */
	/* for(int i=0;i<img_num;i++)
		cout<<gpu_data_test[(map_num-1)*img_num+i]<<endl;
	cout<<endl; */
	/* for(int j=0;j<couts;j++)
			cout<<gpu_min[couts*(map_num-2)+j]<<endl;
		cout<<endl; */
	
	/* for(int j=0;j<couts*map_num;j++)
		cout<<gpu_min[j]<<endl; */
		
	struct timeval startCPU, endCPU;
    gettimeofday(&startCPU, NULL);
	
	for(int i=0;i<map_num;i++)//////回傳比最小
	{
		tem_min=gpu_min[i*couts];tem_index=gpu_index[i*couts];
		for(int j=1;j<min_couts;j++)
		{
			if(tem_min>gpu_min[i*couts+j])
			{
				tem_min=gpu_min[i*couts+j];
				tem_index=gpu_index[i*couts+j];
			}	
		}
		gpu_min[i*couts]=tem_min;
		gpu_index[i*couts]=tem_index;
		if(gpu_min[i*couts]<0.09)//小於門檻值找出比對到的
		{
			gpu_matcher.push_back(i);
			gpu_matcher.push_back(gpu_index[i*couts]);
		}
	}
	
	
		
	gettimeofday(&endCPU, NULL);
    double executime;
    executime = (endCPU.tv_sec - startCPU.tv_sec) * 1000.0;
    executime += (endCPU.tv_usec - startCPU.tv_usec) / 1000.0;
    printf("GPU_CPU find part min time: %13lf msec\n", executime);
	printf("All time: %13lf msec\n", executime+kernelTime+transferTime);
	/////////////////////////////////測試自己創造的值算出的結果///////////////////////////////////////
	#ifdef GPU_Debug_Create
	int block_error=0,index_error=0;
	for(int i=0;i<map_num;i++)
	{
		 if(fabs(gpu_min[i*couts]-0.001)>0.00001)
		{
			//cout<<i<<setw(10)<<gpu_min[i*couts]<<endl;
			block_error++;
		} 
		//cout<<i<<setw(10)<<gpu_min[i]<<endl;
		//cout<<min[couts*i]<<endl;
		//cout<<index[i]<<endl;
	}
	for(int j=0;j<map_num;j++)
		if(gpu_index[j*couts]!=743)
		{
			//cout<<j<<setw(10)<<gpu_index[j*couts]<<endl;
			index_error++;
		}
			
	cout<<	"block_error"<<block_error<<endl;
	cout<<	"index_error"<<index_error<<endl;
	#endif
	/////////////////////////////////測試自己創造的值算出的結果///////////////////////////////////////
    
	// Free device memory
	cudaFree(dev_d_map);
	cudaFree(dev_l_map);
	cudaFree(dev_d_img);
	cudaFree(dev_l_img);
	cudaFree(dev_result);
	cudaFree(dev_min);
	cudaFree(dev_index);
	
	//return min;
	
}


