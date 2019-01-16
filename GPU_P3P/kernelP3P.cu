
#include "kernelP3P.cuh"
//#include "P3P.h
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

__device__ void Ransac(float *mapft_info, float *WL ,float *Rs,float *parm, int *ransac_match_num)
{
	int tid =threadIdx.x;
	int sol_num=parm[4];//u0,vo,fu,fv
	int map_num=parm[5];
	float range=25.0;
	if(blockIdx.x*blockDim.x+tid<sol_num)
	{
		float rx = WL[blockIdx.x*3*blockDim.x+tid*3];
		float ry = WL[blockIdx.x*3*blockDim.x+tid*3+1];
		float rz = WL[blockIdx.x*3*blockDim.x+tid*3+2];
		
		int setsize=0;
		float dist=0;
		for(int i=0;i<map_num;i++)
		{
			float feature[3], h[3];//, h2[3];
			feature[0] = (mapft_info[i*6+3] - parm[0]) / parm[2] * mapft_info[i*6+5];
			feature[1] = (mapft_info[i*6+4] - parm[1]) / parm[3] * mapft_info[i*6+5];
			feature[2] = mapft_info[i*6+5];
			
			h[0] = Rs[blockIdx.x*9*blockDim.x+tid*9] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+3] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+6] * feature[2];
			h[1] = Rs[blockIdx.x*9*blockDim.x+tid*9+1] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+4] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+7] * feature[2];
			h[2] = Rs[blockIdx.x*9*blockDim.x+tid*9+2] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+5] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+8] * feature[2];
			
			dist=0;
			dist = (h[0] - mapft_info[i*6] + rx)*(h[0] - mapft_info[i*6] + rx) +
					(h[1] - mapft_info[i*6+1] + ry)*(h[1] - mapft_info[i*6+1] + ry) +
					(h[2] - mapft_info[i*6+2] + rz)*(h[2] - mapft_info[i*6+2] + rz);
			dist = sqrt(dist);
			if (dist < range)
			{
				setsize++;//符合條件的特徵點個數
			}

		}
		ransac_match_num[blockIdx.x*blockDim.x+tid]=setsize;
	}
}

__device__ void Ransac_dub(double *mapft_info, double *WL ,double *Rs,double *parm, int *ransac_match_num)
{
	int tid =threadIdx.x;
	int sol_num=parm[4];//u0,vo,fu,fv
	int map_num=parm[5];
	double range=25.0;
	if(blockIdx.x*blockDim.x+tid<sol_num)
	{
		double rx = WL[blockIdx.x*3*blockDim.x+tid*3];
		double ry = WL[blockIdx.x*3*blockDim.x+tid*3+1];
		double rz = WL[blockIdx.x*3*blockDim.x+tid*3+2];
		
		int setsize=0;
		double dist=0;
		for(int i=0;i<map_num;i++)
		{
			double feature[3], h[3];//, h2[3];
			feature[0] = (mapft_info[i*6+3] - parm[0]) / parm[2] * mapft_info[i*6+5];
			feature[1] = (mapft_info[i*6+4] - parm[1]) / parm[3] * mapft_info[i*6+5];
			feature[2] = mapft_info[i*6+5];
			
			h[0] = Rs[blockIdx.x*9*blockDim.x+tid*9] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+3] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+6] * feature[2];
			h[1] = Rs[blockIdx.x*9*blockDim.x+tid*9+1] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+4] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+7] * feature[2];
			h[2] = Rs[blockIdx.x*9*blockDim.x+tid*9+2] * feature[0] + Rs[blockIdx.x*9*blockDim.x+tid*9+5] * feature[1] + Rs[blockIdx.x*9*blockDim.x+tid*9+8] * feature[2];
			
			dist=0;
			dist = (h[0] - mapft_info[i*6] + rx)*(h[0] - mapft_info[i*6] + rx) +
					(h[1] - mapft_info[i*6+1] + ry)*(h[1] - mapft_info[i*6+1] + ry) +
					(h[2] - mapft_info[i*6+2] + rz)*(h[2] - mapft_info[i*6+2] + rz);
			dist = sqrt(dist);
			if (dist < range)
			{
				setsize++;//符合條件的特徵點個數
			}

		}
		ransac_match_num[blockIdx.x*blockDim.x+tid]=setsize;
	}
}


__global__ void GPU_P3P(int *matft_NO, int *mapft_NO, float *matft_info, float *mapft_info, int *P3P_NO, float *parm, float *WL ,float *Rs,int *ransac_match_num)
{
	int tid =threadIdx.x;
	//int blockIdx=blockIdx.x;
	int sol_num=parm[4];//u0,vo,fu,fv
	
	if(blockIdx.x*blockDim.x+tid<sol_num)
	{
	
	float ptw[9],pti[6],dep[3];//每個thread有自己的
	int p1,p2,p3;
	
	p1=P3P_NO[blockIdx.x*3*blockDim.x+tid*3];//每個thread取三個編號
	p2=P3P_NO[blockIdx.x*3*blockDim.x+tid*3+1];
	p3=P3P_NO[blockIdx.x*3*blockDim.x+tid*3+2];
	
	ptw[0]=matft_info[p1*6];//第一個點
	ptw[1]=matft_info[p1*6+1];
	ptw[2]=matft_info[p1*6+2];
	pti[0]=matft_info[p1*6+3];
	pti[1]=matft_info[p1*6+4];
	dep[0]=matft_info[p1*6+5]*matft_info[p1*6+5];
	
	ptw[3]=matft_info[p2*6];//第二個點
	ptw[4]=matft_info[p2*6+1];
	ptw[5]=matft_info[p2*6+2];
	pti[2]=matft_info[p2*6+3];
	pti[3]=matft_info[p2*6+4];
	dep[1]=matft_info[p2*6+5]*matft_info[p2*6+5];
	
	ptw[6]=matft_info[p3*6];//第三個點
	ptw[7]=matft_info[p3*6+1];
	ptw[8]=matft_info[p3*6+2];
	pti[4]=matft_info[p3*6+3];
	pti[5]=matft_info[p3*6+4];
	dep[2]=matft_info[p3*6+5]*matft_info[p3*6+5]; 
	
	 float aa,bb,cc;
	aa= sqrt( ((pti[0] - parm[0])/parm[2])*((pti[0] - parm[0])/parm[2])*dep[0] + ((pti[1]-parm[1])/parm[3])*((pti[1]-parm[1])/parm[3])*dep[0] + dep[0]);
	bb = sqrt( ((pti[2] - parm[0])/parm[2])*((pti[2] - parm[0])/parm[2])*dep[1] + ((pti[3]-parm[1])/parm[3])*((pti[3]-parm[1])/parm[3])*dep[1] + dep[1]);
	cc = sqrt( ((pti[4] - parm[0])/parm[2])*((pti[4] - parm[0])/parm[2])*dep[2] + ((pti[5]-parm[1])/parm[3])*((pti[5]-parm[1])/parm[3])*dep[2] + dep[2]);
	 
	/* Rs[blockIdx.x*9*blockDim.x+tid*9]=ptw[0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+1]=ptw[1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+2]=ptw[2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+3]=ptw[3];
	Rs[blockIdx.x*9*blockDim.x+tid*9+4]=ptw[4];
	Rs[blockIdx.x*9*blockDim.x+tid*9+5]=ptw[5];
	Rs[blockIdx.x*9*blockDim.x+tid*9+6]=ptw[6];
	Rs[blockIdx.x*9*blockDim.x+tid*9+7]=ptw[7];
	Rs[blockIdx.x*9*blockDim.x+tid*9+8]=ptw[8];  */
	
	
	//////////////////////////////////////P3P計算///////////////////////////////////
	
	///////////////////////////////算攝影機位置//////////////////////////
	float VAB[3],VAC[3],VBC[3];
		for(int i=0;i<3;i++)
		{
			VAB[i]=ptw[i+3]-ptw[i];//2-1
			VAC[i]=ptw[i+6]-ptw[i];//3-1
			VBC[i]=ptw[i+6]-ptw[i+3];//3-1
		}
		
	//Length of edge between control points
	float Rab,Rac,Rbc;
	Rab=sqrt(VAB[0]*VAB[0]+VAB[1]*VAB[1]+VAB[2]*VAB[2]);
	Rac=sqrt(VAC[0]*VAC[0]+VAC[1]*VAC[1]+VAC[2]*VAC[2]);
	Rbc=sqrt(VBC[0]*VBC[0]+VBC[1]*VBC[1]+VBC[2]*VBC[2]);//後面用不到
	
	float CA[3],CB[3],CC[3];
		CA[0]=(pti[0]-parm[0])/parm[2];	CA[1]=(pti[1]-parm[1])/parm[3];	CA[2]=1;
		CB[0]=(pti[2]-parm[0])/parm[2];	CB[1]=(pti[3]-parm[1])/parm[3];	CB[2]=1;
		CC[0]=(pti[4]-parm[0])/parm[2];	CC[1]=(pti[5]-parm[1])/parm[3];	CC[2]=1;
	
	float RCA,RCB,RCC;
		RCA=sqrt(CA[0]*CA[0]+CA[1]*CA[1]+CA[2]*CA[2]);
		RCB=sqrt(CB[0]*CB[0]+CB[1]*CB[1]+CB[2]*CB[2]);
		RCC=sqrt(CC[0]*CC[0]+CC[1]*CC[1]+CC[2]*CC[2]);
	
	//Normalize
	CA[0]=CA[0]/RCA;	CA[1]=CA[1]/RCA;	CA[2]=CA[2]/RCA;
	CB[0]=CB[0]/RCB;	CB[1]=CB[1]/RCB;	CB[2]=CB[2]/RCB;
	CC[0]=CC[0]/RCC;	CC[1]=CC[1]/RCC;	CC[2]=CC[2]/RCC;
	
	float Rab1,Rac1,Rbc1;//向量長度計算 unit mm
		Rab1=sqrt(CA[0]*CA[0]+CA[1]*CA[1]+CA[2]*CA[2]);
		Rac1=sqrt(CB[0]*CB[0]+CB[1]*CB[1]+CB[2]*CB[2]);
		Rbc1=sqrt(CC[0]*CC[0]+CC[1]*CC[1]+CC[2]*CC[2]);
	
	//Cosine of angles//後面沒用到
	float Calb, Calc, Cblc;
	//Compute Calb Calc Cblc using Law of Cosine
	 Calb = (2 - Rab1*Rab1) / 2;
	Calc = (2 - Rac1*Rac1) / 2;
	Cblc = (2 - Rbc1*Rbc1) / 2; 
	
	//Get cosine of the angles
	float Clab = (aa*aa + Rab*Rab - bb*bb) / (2 * aa*Rab);
	float Clac = (aa*aa + Rac*Rac - cc*cc) / (2 * aa*Rac);
	
	//Get scale along norm vector
	float Raq = aa*Clab;
	float Rap = aa*Clac;
	
	//Get norm vector of plane P1 P2
	float VAB_norm = Rab;
	float VAC_norm = Rac;
	
	float WQ[3],WP[3];
	for (int i = 0; i<3; i++)
	{
		WQ[i]=ptw[i]+ Raq*VAB[i] / VAB_norm;
		WP[i]=ptw[i]+ Rap*VAC[i] / VAC_norm;
	}
	
	//Compute Plane P1 P2 P3
	float NP1[3],NP2[3],NP3[3];
	for (int i = 0; i<3; i++)
	{
		NP1[i]=VAB[i] / VAB_norm;
		NP2[i]=VAC[i] / VAC_norm;
	}
	
	float DP1, DP2, DP3;
	DP1=NP1[0]*WQ[0]+NP1[1]*WQ[1]+NP1[2]*WQ[2];
	DP2=NP2[0]*WP[0]+NP2[1]*WP[1]+NP2[2]*WP[2];
	
	float P1[4],P2[4],P3[4];
	P1[0]=NP1[0];	P1[1]=NP1[1];	P1[2]=NP1[2];	P1[3]=-DP1;
	P2[0]=NP2[0];	P2[1]=NP2[1];	P2[2]=NP2[2];	P2[3]=-DP2;
	
	float VCX[3],VCY[3],VCZ[3];
	for (int i = 0; i<3; i++)
	{
		VCX[i]=CB[i]-CA[i];
		VCY[i]=CC[i]-CA[i];
	}
	
	//計算視線向量外積Z方向的分量判斷使用何種排列(負為AC X AB)
	if ( (VCX[0] * VCY[1] - VCY[0] * VCX[1]) > 0 )	
	{
		//cvCrossProduct(VAC, VAB, NP3);
		NP3[0]=VAC[1]*VAB[2]-VAC[2]*VAB[1];
		NP3[1]=VAC[2]*VAB[0]-VAC[0]*VAB[2];
		NP3[2]=VAC[0]*VAB[1]-VAC[1]*VAB[0];
	}
	else
	{
		//cvCrossProduct(VAB, VAC, NP3);
		NP3[0]=VAB[1]*VAC[2]-VAB[2]*VAC[1];
		NP3[1]=VAB[2]*VAC[0]-VAB[0]*VAC[2];
		NP3[2]=VAB[0]*VAC[1]-VAB[1]*VAC[0];
	}
	
	float NP3_norm=sqrt(NP3[0]*NP3[0]+NP3[1]*NP3[1]+NP3[2]*NP3[2]);
	//Normalize
	NP3[0]=NP3[0]/NP3_norm;	NP3[1]=NP3[1]/NP3_norm;	NP3[2]=NP3[2]/NP3_norm;
	
	DP3=NP3[0]*ptw[0]+NP3[1]*ptw[1]+NP3[2]*ptw[2];//DP3 = cvDotProduct(NP3, W_one);
	
	P3[0]=NP3[0];	P3[1]=NP3[1];	P3[2]=NP3[2];	P3[3]=-DP3;
	
	//克拉馬公式求三平面解 P1 P2 P3
	float delta, delta_x, delta_y, delta_z;
	delta=P1[0]*P2[1]*P3[2]+P2[0]*P3[1]*P1[2]+P3[0]*P1[1]*P2[2]-P3[0]*P2[1]*P1[2]-P1[0]*P3[1]*P2[2]-P2[0]*P1[1]*P3[2];
	delta_x=P2[1]*P3[2]*P1[3]+P1[1]*P2[2]*P3[3]+P3[1]*P2[3]*P1[2]-P2[1]*P1[2]*P3[3]-P1[1]*P2[3]*P3[2]-P3[1]*P1[3]*P2[2];
	delta_y=P1[0]*P3[2]*P2[3]+P2[0]*P1[2]*P3[3]+P3[0]*P2[2]*P1[3]-P3[0]*P1[2]*P2[3]-P1[0]*P2[2]*P3[3]-P2[0]*P3[2]*P1[3];
	delta_z=P1[0]*P2[1]*P3[3]+P2[0]*P3[1]*P1[3]+P3[0]*P1[1]*P2[3]-P3[0]*P2[1]*P1[3]-P1[0]*P3[1]*P2[3]-P2[0]*P1[1]*P3[3];
	
	float WR[3];
	if(delta!=0)
	{
		//上面係數移項，因此差一個負號
		WR[0]=-delta_x/delta;	WR[1]=-delta_y/delta;	WR[2]=-delta_z/delta;
	}
	else
	{
		WR[0]=1000000;	WR[1]=1000000;	WR[2]=1000000;
	}
	
	
	//Get length of LR
	float Rar, Rlr;
	Rar=sqrt((ptw[0]-WR[0])*(ptw[0]-WR[0])+(ptw[1]-WR[1])*(ptw[1]-WR[1])+(ptw[2]-WR[2])*(ptw[2]-WR[2]));
	if(aa*aa - Rar*Rar>0)
		Rlr = sqrt(aa*aa - Rar*Rar);//當aa*aa - Rar*Rar<0時，Rlr則為nan
	else
		Rlr = 10e6;
	//Get Position of L in world frame
	//WL=WR+NP3*Rlr;
	float WL_test[3];
	WL_test[0]=WR[0]+NP3[0]*Rlr;
	WL_test[1]=WR[1]+NP3[1]*Rlr;
	WL_test[2]=WR[2]+NP3[2]*Rlr;
	
	WL[blockIdx.x*3*blockDim.x+tid*3]=WL_test[0];
	WL[blockIdx.x*3*blockDim.x+tid*3+1]=WL_test[1];
	WL[blockIdx.x*3*blockDim.x+tid*3+2]=WL_test[2];
	
	///////////////////////////////////////建坐標系算旋轉矩陣/////////////////////////////////////////////
	VCZ[0]=VCX[1]*VCY[2]-VCX[2]*VCY[1];	//cvCrossProduct(VCX, VCY, VCZ);
	VCZ[1]=VCX[2]*VCY[0]-VCX[0]*VCY[2];
	VCZ[2]=VCX[0]*VCY[1]-VCX[1]*VCY[0];
	
	VCY[0]=VCZ[1]*VCX[2]-VCZ[2]*VCX[1];	//cvCrossProduct(VCZ, VCX, VCY);
	VCY[1]=VCZ[2]*VCX[0]-VCZ[0]*VCX[2];
	VCY[2]=VCZ[0]*VCX[1]-VCZ[1]*VCX[0];
	
	//Normalize
	float VCX_norm,VCY_norm,VCZ_norm;
	VCX_norm=sqrt(VCX[0]*VCX[0]+VCX[1]*VCX[1]+VCX[2]*VCX[2]);
	VCY_norm=sqrt(VCY[0]*VCY[0]+VCY[1]*VCY[1]+VCY[2]*VCY[2]);
	VCZ_norm=sqrt(VCZ[0]*VCZ[0]+VCZ[1]*VCZ[1]+VCZ[2]*VCZ[2]);
	for(int i=0;i<3;i++)
	{
		VCX[i]=VCX[i]/VCX_norm;	VCY[i]=VCY[i]/VCY_norm;	VCZ[i]=VCZ[i]/VCZ_norm;
	}
	
	//%Get ray in the world frame
	float Vla[3],Vlb[3],Vlc[3];
	for(int i=0;i<3;i++)
	{
		Vla[i]=ptw[i]-WL_test[i];	//Vla=WA-WL;
		Vlb[i]=ptw[i+3]-WL_test[i];	//Vlb=WB-WL;
		Vlc[i]=ptw[i+6]-WL_test[i];	//Vlc=WC-WL;
	}
	
	//Normalize
	float Vla_norm,Vlb_norm,Vlc_norm;
	Vla_norm=sqrt(Vla[0]*Vla[0]+Vla[1]*Vla[1]+Vla[2]*Vla[2]);
	Vlb_norm=sqrt(Vlb[0]*Vlb[0]+Vlb[1]*Vlb[1]+Vlb[2]*Vlb[2]);
	Vlc_norm=sqrt(Vlc[0]*Vlc[0]+Vlc[1]*Vlc[1]+Vlc[2]*Vlc[2]);
	for(int i=0;i<3;i++)
	{
		Vla[i]=Vla[i]/Vla_norm;	Vlb[i]=Vlb[i]/Vlb_norm;	Vlc[i]=Vlc[i]/Vlc_norm;
	}
	
	float WA1[3],WB1[3],WC1[3];
	for(int i=0;i<3;i++)
	{
		WA1[i]=WL_test[i]+Vla[i];	//WA1=WL+1*Vla;
		WB1[i]=WL_test[i]+Vlb[i];	//WB1=WL+1*Vlb;
		WC1[i]=WL_test[i]+Vlc[i];	//WC1=WL+1*Vlc;
	}
	
	float vcx[3],vcy[3],vcz[3];
	for(int i=0;i<3;i++)
	{
		vcx[i]=WB1[i]-WA1[i];	//vcx=WB1-WA1;
		vcy[i]=WC1[i]-WA1[i];	//vcy=WC1-WA1;
	}
	
	
	vcz[0]=vcx[1]*vcy[2]-vcx[2]*vcy[1];	//cvCrossProduct(vcx, vcy, vcz);
	vcz[1]=vcx[2]*vcy[0]-vcx[0]*vcy[2];
	vcz[2]=vcx[0]*vcy[1]-vcx[1]*vcy[0];
	
	vcy[0]=vcz[1]*vcx[2]-vcz[2]*vcx[1];	//cvCrossProduct(vcz, vcx, vcy);
	vcy[1]=vcz[2]*vcx[0]-vcz[0]*vcx[2];
	vcy[2]=vcz[0]*vcx[1]-vcz[1]*vcx[0];
	
	
	
	//Normalize
	//float vcx_norm_test,vcy_norm_test,vcz_norm_test;
	float vcx_norm,vcy_norm,vcz_norm;
	
	/* vcx_norm_test=vcx[0]*vcx[0]+vcx[1]*vcx[1]+vcx[2]*vcx[2];
	vcy_norm_test=vcy[0]*vcy[0]+vcy[1]*vcy[1]+vcy[2]*vcy[2];
	vcz_norm_test=vcz[0]*vcz[0]+vcz[1]*vcz[1]+vcz[2]*vcz[2];
	
	if(vcx_norm_test==0)
		vcx_norm=sqrt(10e-12);
	else	vcx_norm=sqrt(vcx_norm_test);
	
	if(vcy_norm_test==0)
		vcy_norm_test=sqrt(10e-12);
	else	vcy_norm=sqrt(vcy_norm_test);
	
	if(vcz_norm_test==0)
		vcz_norm_test=sqrt(10e-12);
	else	vcz_norm=sqrt(vcz_norm_test); */
	
	vcx_norm=sqrt(vcx[0]*vcx[0]+vcx[1]*vcx[1]+vcx[2]*vcx[2]);
	vcy_norm=sqrt(vcy[0]*vcy[0]+vcy[1]*vcy[1]+vcy[2]*vcy[2]);
	vcz_norm=sqrt(vcz[0]*vcz[0]+vcz[1]*vcz[1]+vcz[2]*vcz[2]);
	
	for(int i=0;i<3;i++)
	{
		vcx[i]=vcx[i]/vcx_norm;	vcy[i]=vcy[i]/vcy_norm;	vcz[i]=vcz[i]/vcz_norm;
	}
	
	//R=[VCx VCy VCz]*inv([vcx vcy vcz]);
	float R1[3][3],R2[3][3]/*,R3[3][3]*/;
	for(int i=0;i<3;i++)
	{
		R1[i][0]=VCX[i];	R1[i][1]=VCY[i];	R1[i][2]=VCZ[i];
		R2[0][i]=vcx[i];	R2[1][i]=vcy[i];	R2[2][i]=vcz[i];	//因為之後要cvTranspose(R2, R2);所以放的時候直接
	}
	
	//矩陣相乘(3*3)R1*R2
	float Rext[3][3];
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			float sum=0.0;
			for(int k=0;k<3;k++)
				sum+=R1[i][k]*R2[k][j];
			Rext[i][j]=sum;
		}
	}
		
	
	Rs[blockIdx.x*9*blockDim.x+tid*9]=Rext[0][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+1]=Rext[0][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+2]=Rext[0][2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+3]=Rext[1][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+4]=Rext[1][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+5]=Rext[1][2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+6]=Rext[2][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+7]=Rext[2][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+8]=Rext[2][2]; 
		
	//for (int k = 0; k< 3; k++)
	//{
	//	cvmSet(text, k, 0, -cvmGet(WL, k, 0));
	//}
	//cvMatMul(Rext, text, text);			
	/* for(int i=0;i<3;i++)		text目前沒用到
	{
		float sum=0.0;
		for(int k=0;k<3;k++)
			sum+=Rext[i][k]*(-WL_test[k]);
		text[i]=sum;
	} */

	}
	
	Ransac(mapft_info, WL ,Rs,parm, ransac_match_num);
}

__global__ void GPU_P3P_dub(int *matft_NO, int *mapft_NO, double *matft_info, double *mapft_info, int *P3P_NO, double *parm, double *WL ,double *Rs,int *ransac_match_num)
{
	int tid =threadIdx.x;
	//int blockIdx=blockIdx.x;
	int sol_num=parm[4];//u0,vo,fu,fv
	
	if(blockIdx.x*blockDim.x+tid<sol_num)
	{
	
	double ptw[9],pti[6],dep[3];//每個thread有自己的
	int p1,p2,p3;
	
	p1=P3P_NO[blockIdx.x*3*blockDim.x+tid*3];//每個thread取三個編號
	p2=P3P_NO[blockIdx.x*3*blockDim.x+tid*3+1];
	p3=P3P_NO[blockIdx.x*3*blockDim.x+tid*3+2];
	
	ptw[0]=matft_info[p1*6];//第一個點
	ptw[1]=matft_info[p1*6+1];
	ptw[2]=matft_info[p1*6+2];
	pti[0]=matft_info[p1*6+3];
	pti[1]=matft_info[p1*6+4];
	dep[0]=matft_info[p1*6+5]*matft_info[p1*6+5];
	
	ptw[3]=matft_info[p2*6];//第二個點
	ptw[4]=matft_info[p2*6+1];
	ptw[5]=matft_info[p2*6+2];
	pti[2]=matft_info[p2*6+3];
	pti[3]=matft_info[p2*6+4];
	dep[1]=matft_info[p2*6+5]*matft_info[p2*6+5];
	
	ptw[6]=matft_info[p3*6];//第三個點
	ptw[7]=matft_info[p3*6+1];
	ptw[8]=matft_info[p3*6+2];
	pti[4]=matft_info[p3*6+3];
	pti[5]=matft_info[p3*6+4];
	dep[2]=matft_info[p3*6+5]*matft_info[p3*6+5]; 
	
	 double aa,bb,cc;
	aa= sqrt( ((pti[0] - parm[0])/parm[2])*((pti[0] - parm[0])/parm[2])*dep[0] + ((pti[1]-parm[1])/parm[3])*((pti[1]-parm[1])/parm[3])*dep[0] + dep[0]);
	bb = sqrt( ((pti[2] - parm[0])/parm[2])*((pti[2] - parm[0])/parm[2])*dep[1] + ((pti[3]-parm[1])/parm[3])*((pti[3]-parm[1])/parm[3])*dep[1] + dep[1]);
	cc = sqrt( ((pti[4] - parm[0])/parm[2])*((pti[4] - parm[0])/parm[2])*dep[2] + ((pti[5]-parm[1])/parm[3])*((pti[5]-parm[1])/parm[3])*dep[2] + dep[2]);
	 
	/* Rs[blockIdx.x*9*blockDim.x+tid*9]=ptw[0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+1]=ptw[1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+2]=ptw[2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+3]=ptw[3];
	Rs[blockIdx.x*9*blockDim.x+tid*9+4]=ptw[4];
	Rs[blockIdx.x*9*blockDim.x+tid*9+5]=ptw[5];
	Rs[blockIdx.x*9*blockDim.x+tid*9+6]=ptw[6];
	Rs[blockIdx.x*9*blockDim.x+tid*9+7]=ptw[7];
	Rs[blockIdx.x*9*blockDim.x+tid*9+8]=ptw[8];  */
	
	
	//////////////////////////////////////P3P計算///////////////////////////////////
	
	///////////////////////////////算攝影機位置//////////////////////////
	double VAB[3],VAC[3],VBC[3];
		for(int i=0;i<3;i++)
		{
			VAB[i]=ptw[i+3]-ptw[i];//2-1
			VAC[i]=ptw[i+6]-ptw[i];//3-1
			VBC[i]=ptw[i+6]-ptw[i+3];//3-1
		}
		
	//Length of edge between control points
	double Rab,Rac,Rbc;
	Rab=sqrt(VAB[0]*VAB[0]+VAB[1]*VAB[1]+VAB[2]*VAB[2]);
	Rac=sqrt(VAC[0]*VAC[0]+VAC[1]*VAC[1]+VAC[2]*VAC[2]);
	Rbc=sqrt(VBC[0]*VBC[0]+VBC[1]*VBC[1]+VBC[2]*VBC[2]);//後面用不到
	
	double CA[3],CB[3],CC[3];
		CA[0]=(pti[0]-parm[0])/parm[2];	CA[1]=(pti[1]-parm[1])/parm[3];	CA[2]=1;
		CB[0]=(pti[2]-parm[0])/parm[2];	CB[1]=(pti[3]-parm[1])/parm[3];	CB[2]=1;
		CC[0]=(pti[4]-parm[0])/parm[2];	CC[1]=(pti[5]-parm[1])/parm[3];	CC[2]=1;
	
	double RCA,RCB,RCC;
		RCA=sqrt(CA[0]*CA[0]+CA[1]*CA[1]+CA[2]*CA[2]);
		RCB=sqrt(CB[0]*CB[0]+CB[1]*CB[1]+CB[2]*CB[2]);
		RCC=sqrt(CC[0]*CC[0]+CC[1]*CC[1]+CC[2]*CC[2]);
	
	//Normalize
	CA[0]=CA[0]/RCA;	CA[1]=CA[1]/RCA;	CA[2]=CA[2]/RCA;
	CB[0]=CB[0]/RCB;	CB[1]=CB[1]/RCB;	CB[2]=CB[2]/RCB;
	CC[0]=CC[0]/RCC;	CC[1]=CC[1]/RCC;	CC[2]=CC[2]/RCC;
	
		
	double Rab1,Rac1,Rbc1;//向量長度計算 unit mm
		Rab1=sqrt(CA[0]*CA[0]+CA[1]*CA[1]+CA[2]*CA[2]);
		Rac1=sqrt(CB[0]*CB[0]+CB[1]*CB[1]+CB[2]*CB[2]);
		Rbc1=sqrt(CC[0]*CC[0]+CC[1]*CC[1]+CC[2]*CC[2]);
	
	
	//Cosine of angles//後面沒用到
	double Calb, Calc, Cblc;
	//Compute Calb Calc Cblc using Law of Cosine
	 Calb = (2 - Rab1*Rab1) / 2;
	Calc = (2 - Rac1*Rac1) / 2;
	Cblc = (2 - Rbc1*Rbc1) / 2; 
	
	//Get cosine of the angles
	double Clab = (aa*aa + Rab*Rab - bb*bb) / (2 * aa*Rab);
	double Clac = (aa*aa + Rac*Rac - cc*cc) / (2 * aa*Rac);
	
	//Get scale along norm vector
	double Raq = aa*Clab;
	double Rap = aa*Clac;
	
		
	//Get norm vector of plane P1 P2
	double VAB_norm = Rab;
	double VAC_norm = Rac;
	
	double WQ[3],WP[3];
	for (int i = 0; i<3; i++)
	{
		WQ[i]=ptw[i]+ Raq*VAB[i] / VAB_norm;
		WP[i]=ptw[i]+ Rap*VAC[i] / VAC_norm;
	}
	
	//Compute Plane P1 P2 P3
	double NP1[3],NP2[3],NP3[3];
	for (int i = 0; i<3; i++)
	{
		NP1[i]=VAB[i] / VAB_norm;
		NP2[i]=VAC[i] / VAC_norm;
	}
	
	double DP1, DP2, DP3;
	DP1=NP1[0]*WQ[0]+NP1[1]*WQ[1]+NP1[2]*WQ[2];
	DP2=NP2[0]*WP[0]+NP2[1]*WP[1]+NP2[2]*WP[2];
	
	double P1[4],P2[4],P3[4];
	P1[0]=NP1[0];	P1[1]=NP1[1];	P1[2]=NP1[2];	P1[3]=-DP1;
	P2[0]=NP2[0];	P2[1]=NP2[1];	P2[2]=NP2[2];	P2[3]=-DP2;
	
	double VCX[3],VCY[3],VCZ[3];
	for (int i = 0; i<3; i++)
	{
		VCX[i]=CB[i]-CA[i];
		VCY[i]=CC[i]-CA[i];
	}
	
	//計算視線向量外積Z方向的分量判斷使用何種排列(負為AC X AB)
	if ( (VCX[0] * VCY[1] - VCY[0] * VCX[1]) > 0 )	
	{
		//cvCrossProduct(VAC, VAB, NP3);
		NP3[0]=VAC[1]*VAB[2]-VAC[2]*VAB[1];
		NP3[1]=VAC[2]*VAB[0]-VAC[0]*VAB[2];
		NP3[2]=VAC[0]*VAB[1]-VAC[1]*VAB[0];
	}
	else
	{
		//cvCrossProduct(VAB, VAC, NP3);
		NP3[0]=VAB[1]*VAC[2]-VAB[2]*VAC[1];
		NP3[1]=VAB[2]*VAC[0]-VAB[0]*VAC[2];
		NP3[2]=VAB[0]*VAC[1]-VAB[1]*VAC[0];
	}
	
	double NP3_norm=sqrt(NP3[0]*NP3[0]+NP3[1]*NP3[1]+NP3[2]*NP3[2]);
	//Normalize
	NP3[0]=NP3[0]/NP3_norm;	NP3[1]=NP3[1]/NP3_norm;	NP3[2]=NP3[2]/NP3_norm;
	
	DP3=NP3[0]*ptw[0]+NP3[1]*ptw[1]+NP3[2]*ptw[2];//DP3 = cvDotProduct(NP3, W_one);
	
	P3[0]=NP3[0];	P3[1]=NP3[1];	P3[2]=NP3[2];	P3[3]=-DP3;
	
	//克拉馬公式求三平面解 P1 P2 P3
	double delta, delta_x, delta_y, delta_z;
	delta=P1[0]*P2[1]*P3[2]+P2[0]*P3[1]*P1[2]+P3[0]*P1[1]*P2[2]-P3[0]*P2[1]*P1[2]-P1[0]*P3[1]*P2[2]-P2[0]*P1[1]*P3[2];
	delta_x=P2[1]*P3[2]*P1[3]+P1[1]*P2[2]*P3[3]+P3[1]*P2[3]*P1[2]-P2[1]*P1[2]*P3[3]-P1[1]*P2[3]*P3[2]-P3[1]*P1[3]*P2[2];
	delta_y=P1[0]*P3[2]*P2[3]+P2[0]*P1[2]*P3[3]+P3[0]*P2[2]*P1[3]-P3[0]*P1[2]*P2[3]-P1[0]*P2[2]*P3[3]-P2[0]*P3[2]*P1[3];
	delta_z=P1[0]*P2[1]*P3[3]+P2[0]*P3[1]*P1[3]+P3[0]*P1[1]*P2[3]-P3[0]*P2[1]*P1[3]-P1[0]*P3[1]*P2[3]-P2[0]*P1[1]*P3[3];
	
	double WR[3];
	if(delta!=0)
	{
		//上面係數移項，因此差一個負號
		WR[0]=-delta_x/delta;	WR[1]=-delta_y/delta;	WR[2]=-delta_z/delta;
	}
	else
	{
		WR[0]=1000000;	WR[1]=1000000;	WR[2]=1000000;
	}
	
	
	//Get length of LR
	double Rar, Rlr;
	Rar=sqrt((ptw[0]-WR[0])*(ptw[0]-WR[0])+(ptw[1]-WR[1])*(ptw[1]-WR[1])+(ptw[2]-WR[2])*(ptw[2]-WR[2]));
	if(aa*aa - Rar*Rar>0)
		Rlr = sqrt(aa*aa - Rar*Rar);//當aa*aa - Rar*Rar<0時，Rlr則為nan
	else
		Rlr = 10e6;
	//Get Position of L in world frame
	//WL=WR+NP3*Rlr;
	double WL_test[3];
	WL_test[0]=WR[0]+NP3[0]*Rlr;
	WL_test[1]=WR[1]+NP3[1]*Rlr;
	WL_test[2]=WR[2]+NP3[2]*Rlr;
	
	WL[blockIdx.x*3*blockDim.x+tid*3]=WL_test[0];
	WL[blockIdx.x*3*blockDim.x+tid*3+1]=WL_test[1];
	WL[blockIdx.x*3*blockDim.x+tid*3+2]=WL_test[2];
	
	///////////////////////////////////////建坐標系算旋轉矩陣/////////////////////////////////////////////
	VCZ[0]=VCX[1]*VCY[2]-VCX[2]*VCY[1];	//cvCrossProduct(VCX, VCY, VCZ);
	VCZ[1]=VCX[2]*VCY[0]-VCX[0]*VCY[2];
	VCZ[2]=VCX[0]*VCY[1]-VCX[1]*VCY[0];
	
	VCY[0]=VCZ[1]*VCX[2]-VCZ[2]*VCX[1];	//cvCrossProduct(VCZ, VCX, VCY);
	VCY[1]=VCZ[2]*VCX[0]-VCZ[0]*VCX[2];
	VCY[2]=VCZ[0]*VCX[1]-VCZ[1]*VCX[0];
	
	//Normalize
	double VCX_norm,VCY_norm,VCZ_norm;
	VCX_norm=sqrt(VCX[0]*VCX[0]+VCX[1]*VCX[1]+VCX[2]*VCX[2]);
	VCY_norm=sqrt(VCY[0]*VCY[0]+VCY[1]*VCY[1]+VCY[2]*VCY[2]);
	VCZ_norm=sqrt(VCZ[0]*VCZ[0]+VCZ[1]*VCZ[1]+VCZ[2]*VCZ[2]);
	for(int i=0;i<3;i++)
	{
		VCX[i]=VCX[i]/VCX_norm;	VCY[i]=VCY[i]/VCY_norm;	VCZ[i]=VCZ[i]/VCZ_norm;
	}
	
	//%Get ray in the world frame
	double Vla[3],Vlb[3],Vlc[3];
	for(int i=0;i<3;i++)
	{
		Vla[i]=ptw[i]-WL_test[i];	//Vla=WA-WL;
		Vlb[i]=ptw[i+3]-WL_test[i];	//Vlb=WB-WL;
		Vlc[i]=ptw[i+6]-WL_test[i];	//Vlc=WC-WL;
	}
	
	//Normalize
	double Vla_norm,Vlb_norm,Vlc_norm;
	Vla_norm=sqrt(Vla[0]*Vla[0]+Vla[1]*Vla[1]+Vla[2]*Vla[2]);
	Vlb_norm=sqrt(Vlb[0]*Vlb[0]+Vlb[1]*Vlb[1]+Vlb[2]*Vlb[2]);
	Vlc_norm=sqrt(Vlc[0]*Vlc[0]+Vlc[1]*Vlc[1]+Vlc[2]*Vlc[2]);
	for(int i=0;i<3;i++)
	{
		Vla[i]=Vla[i]/Vla_norm;	Vlb[i]=Vlb[i]/Vlb_norm;	Vlc[i]=Vlc[i]/Vlc_norm;
	}
	
	double WA1[3],WB1[3],WC1[3];
	for(int i=0;i<3;i++)
	{
		WA1[i]=WL_test[i]+Vla[i];	//WA1=WL+1*Vla;
		WB1[i]=WL_test[i]+Vlb[i];	//WB1=WL+1*Vlb;
		WC1[i]=WL_test[i]+Vlc[i];	//WC1=WL+1*Vlc;
	}
	
	double vcx[3],vcy[3],vcz[3];
	for(int i=0;i<3;i++)
	{
		vcx[i]=WB1[i]-WA1[i];	//vcx=WB1-WA1;
		vcy[i]=WC1[i]-WA1[i];	//vcy=WC1-WA1;
	}
	
	
	vcz[0]=vcx[1]*vcy[2]-vcx[2]*vcy[1];	//cvCrossProduct(vcx, vcy, vcz);
	vcz[1]=vcx[2]*vcy[0]-vcx[0]*vcy[2];
	vcz[2]=vcx[0]*vcy[1]-vcx[1]*vcy[0];
	
	vcy[0]=vcz[1]*vcx[2]-vcz[2]*vcx[1];	//cvCrossProduct(vcz, vcx, vcy);
	vcy[1]=vcz[2]*vcx[0]-vcz[0]*vcx[2];
	vcy[2]=vcz[0]*vcx[1]-vcz[1]*vcx[0];
	
	
	
	//Normalize
	//float vcx_norm_test,vcy_norm_test,vcz_norm_test;
	double vcx_norm,vcy_norm,vcz_norm;
	
	/* vcx_norm_test=vcx[0]*vcx[0]+vcx[1]*vcx[1]+vcx[2]*vcx[2];
	vcy_norm_test=vcy[0]*vcy[0]+vcy[1]*vcy[1]+vcy[2]*vcy[2];
	vcz_norm_test=vcz[0]*vcz[0]+vcz[1]*vcz[1]+vcz[2]*vcz[2];
	
	if(vcx_norm_test==0)
		vcx_norm=sqrt(10e-12);
	else	vcx_norm=sqrt(vcx_norm_test);
	
	if(vcy_norm_test==0)
		vcy_norm_test=sqrt(10e-12);
	else	vcy_norm=sqrt(vcy_norm_test);
	
	if(vcz_norm_test==0)
		vcz_norm_test=sqrt(10e-12);
	else	vcz_norm=sqrt(vcz_norm_test); */
	
	vcx_norm=sqrt(vcx[0]*vcx[0]+vcx[1]*vcx[1]+vcx[2]*vcx[2]);
	vcy_norm=sqrt(vcy[0]*vcy[0]+vcy[1]*vcy[1]+vcy[2]*vcy[2]);
	vcz_norm=sqrt(vcz[0]*vcz[0]+vcz[1]*vcz[1]+vcz[2]*vcz[2]);
	
	for(int i=0;i<3;i++)
	{
		vcx[i]=vcx[i]/vcx_norm;	vcy[i]=vcy[i]/vcy_norm;	vcz[i]=vcz[i]/vcz_norm;
	}
	
	//R=[VCx VCy VCz]*inv([vcx vcy vcz]);
	double R1[3][3],R2[3][3]/*,R3[3][3]*/;
	for(int i=0;i<3;i++)
	{
		R1[i][0]=VCX[i];	R1[i][1]=VCY[i];	R1[i][2]=VCZ[i];
		R2[0][i]=vcx[i];	R2[1][i]=vcy[i];	R2[2][i]=vcz[i];	//因為之後要cvTranspose(R2, R2);所以放的時候直接
	}
	
	//矩陣相乘(3*3)R1*R2
	double Rext[3][3];
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			double sum=0.0;
			for(int k=0;k<3;k++)
				sum+=R1[i][k]*R2[k][j];
			Rext[i][j]=sum;
		}
	}
		
	
	Rs[blockIdx.x*9*blockDim.x+tid*9]=Rext[0][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+1]=Rext[0][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+2]=Rext[0][2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+3]=Rext[1][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+4]=Rext[1][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+5]=Rext[1][2];
	Rs[blockIdx.x*9*blockDim.x+tid*9+6]=Rext[2][0];
	Rs[blockIdx.x*9*blockDim.x+tid*9+7]=Rext[2][1];
	Rs[blockIdx.x*9*blockDim.x+tid*9+8]=Rext[2][2]; 
		
	//for (int k = 0; k< 3; k++)
	//{
	//	cvmSet(text, k, 0, -cvmGet(WL, k, 0));
	//}
	//cvMatMul(Rext, text, text);			
	/* for(int i=0;i<3;i++)		text目前沒用到
	{
		float sum=0.0;
		for(int k=0;k<3;k++)
			sum+=Rext[i][k]*(-WL_test[k]);
		text[i]=sum;
	} */

	}
	
	Ransac_dub(mapft_info, WL ,Rs,parm, ransac_match_num);
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

void launchVOKernel(std::vector<keepfeature> matchfeature , std::vector<keepfeature> map_feature,std::vector<Ransac_Pos> &ransac_match,float *GPU_WL,float *GPU_Rs,int *ransac_match_num,double l_u0,double l_v0,double l_fu,double l_fv)
{
	int thredperblock = NT;
	int sol_num=(matchfeature.size()*(matchfeature.size()-1)*(matchfeature.size()-2))/6; //n取3
	int blocknum=(sol_num+thredperblock-1)/thredperblock;
	/////////////////// allocate memory on the cpu side//////////////////////
	int *matchfeature_NO,*map_feature_NO;//編號
	float *matchfeature_info, *map_feature_info;
	int *P3P_NO;//建model的特徵點編號
	//float *WL, *Rs;
	float *parm;
	//int *ransac_match_num;//符合共識集合的數量
	matchfeature_NO = (int*)malloc(matchfeature.size()* sizeof(int));
	map_feature_NO = (int*)malloc(map_feature.size()* sizeof(int));
	matchfeature_info = (float*)malloc(6 *matchfeature.size()* sizeof(float));//hx hy hz ix iy
	map_feature_info = (float*)malloc(6 *map_feature.size()* sizeof(float));//hx hy hz ix iy
	//WL=(float*)malloc(3 *sol_num* sizeof(float));//x y z
	//Rs=(float*)malloc(9 *sol_num* sizeof(float));//11 12 13 21 22 23 31 32 33
	P3P_NO=(int*)malloc(3 *sol_num* sizeof(int));
	parm=(float*)malloc(6 * sizeof(float));//l_u0,l_v0,l_fu,l_fv,map_feature.size()
	//ransac_match_num=(int*)malloc(sol_num* sizeof(int));
	////////////////// put data in the host memory////////////////////////
	int times=0;
	for(int g=0;g<matchfeature.size()-2;g++)
	{
		for(int g_two=g+1 ;  g_two< matchfeature.size()-1 ; g_two++ )
		{
			for(int g_three=g_two+1 ;  g_three< matchfeature.size() ; g_three++ )
			{
				P3P_NO[times]=g;
				P3P_NO[times+1]=g_two;
				P3P_NO[times+2]=g_three;
				times+=3;
			}
		}
	}
	/* for(int i=0;i<3*sol_num;i+=3)
	{
		cout<<P3P_NO[i]<<setw(5)<<P3P_NO[i+1]<<setw(5)<<P3P_NO[i+2]<<endl;
	} */
	parm[0]=l_u0;parm[1]=l_v0;parm[2]=l_fu;parm[3]=l_fv;parm[4]=sol_num;parm[5]=map_feature.size();
	
	for(int i=0;i<matchfeature.size();i++)
	{
		matchfeature_NO[i]=matchfeature[i].num;
		matchfeature_info[i*6]=matchfeature[i].hx;
		matchfeature_info[i*6+1]=matchfeature[i].hy;
		matchfeature_info[i*6+2]=matchfeature[i].hz;
		matchfeature_info[i*6+3]=matchfeature[i].l_ix;
		matchfeature_info[i*6+4]=matchfeature[i].l_iy;
		matchfeature_info[i*6+5]=matchfeature[i].depthvalue;
	}

	for(int j=0;j<map_feature.size();j++)
	{
		map_feature_NO[j]=map_feature[j].num;
		map_feature_info[j*6]=map_feature[j].hx;
		map_feature_info[j*6+1]=map_feature[j].hy;
		map_feature_info[j*6+2]=map_feature[j].hz;
		map_feature_info[j*6+3]=map_feature[j].l_ix;
		map_feature_info[j*6+4]=map_feature[j].l_iy;
		map_feature_info[j*6+5]=map_feature[j].depthvalue;
	}
		
	////////////////// allocate the memory on the GPU ///////////////////////
	
	int *dev_matchfeature_NO,*dev_map_feature_NO;//編號
	float *dev_matchfeature_info, *dev_map_feature_info;
	float *dev_WL,*dev_Rs;
	int *dev_P3P_NO;//建model的特徵點編號
	float *dev_parm;
	int *dev_ransac_match_num;//符合共識集合的數量
	cudaMalloc((void **)&dev_matchfeature_NO, matchfeature.size()* sizeof(int));
	cudaMalloc((void **)&dev_map_feature_NO, map_feature.size()* sizeof(int));
	cudaMalloc((void **)&dev_matchfeature_info, 6 *matchfeature.size()* sizeof(float));
	cudaMalloc((void **)&dev_map_feature_info, 6 *map_feature.size()* sizeof(float));
	cudaMalloc((void **)&dev_WL, 3 *sol_num* sizeof(float));
	cudaMalloc((void **)&dev_Rs, 9 *sol_num* sizeof(float));
	cudaMalloc((void **)&dev_P3P_NO, 3 *sol_num* sizeof(int));
	cudaMalloc((void **)&dev_parm, 6*sizeof(float));
	cudaMalloc((void **)&dev_ransac_match_num, sol_num* sizeof(int));
	////////////////// copy the arrays to the GPU/////////////////////////
	cudaMemcpy(dev_matchfeature_NO, matchfeature_NO, matchfeature.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_map_feature_NO, map_feature_NO, map_feature.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matchfeature_info, matchfeature_info, 6 *matchfeature.size()* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_map_feature_info, map_feature_info, 6 *map_feature.size()* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_WL, GPU_WL, 3 *sol_num* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Rs, GPU_Rs, 9 *sol_num* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_P3P_NO, P3P_NO, 3 *sol_num* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_parm, parm, 6* sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ransac_match_num, ransac_match_num, sol_num* sizeof(int), cudaMemcpyHostToDevice);
	
	cout<<"called launchVOKernel ok"<<endl;
	cout<<"blocknum"<<setw(10)<<blocknum<<endl;
	cout<<"thredperblock"<<setw(10)<<thredperblock<<endl;
	
	// Get start time event
	cudaEvent_t start_kernel, stop_kernel;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventRecord(start_kernel, 0);
	
	GPU_P3P << < blocknum, thredperblock >> > (dev_matchfeature_NO,dev_map_feature_NO,dev_matchfeature_info,dev_map_feature_info,dev_P3P_NO,dev_parm,dev_WL,dev_Rs,dev_ransac_match_num);

	
	// Get stop time event
	cudaEventRecord(stop_kernel, 0);
	cudaEventSynchronize(stop_kernel);
	// Compute execution time
	float kernelTime;
	cudaEventElapsedTime(&kernelTime, start_kernel, stop_kernel);
	printf("GPU float time: %13f msec\n", kernelTime);
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
	
	cudaMemcpy(GPU_WL, dev_WL, 3 *sol_num* sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(GPU_Rs, dev_Rs, 9 *sol_num* sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ransac_match_num, dev_ransac_match_num, sol_num* sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop_tra, 0);
	cudaEventSynchronize(stop_tra);

	// Compute execution time
	float transferTime;
	cudaEventElapsedTime(&transferTime, start_tra, stop_tra);
	printf("GPU transfer time: %13f msec\n", transferTime);
	cout<<"---------------------------------------------------"<<endl;
	cudaEventDestroy(start_tra);
	cudaEventDestroy(stop_tra);
	
	
	
	/* cout<<"-----------------------GPU_ransac_match--------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		cout<<ransac_match_num[i]<<endl;
	}
	cout<<"-----------------------GPU_ransac_match--------------------------"<<endl;
	cout<<"sol num "<<setw(5)<<sol_num<<endl; */
	
	/* cout<<"-----------------------GPU sol camera pose----------------------------"<<endl; 
	for(int i=0;i<sol_num;i++)
	{
		cout<<GPU_WL[i*3]<<setw(15)<<GPU_WL[i*3+1]<<setw(15)<<GPU_WL[i*3+2]<<endl;
	}
	cout<<"-----------------------GPU sol rotation matrix----------------------------"<<endl; 
	for(int i=0;i<sol_num;i++)
	{
		cout<<Rs[i*9]<<setw(15)<<Rs[i*9+1]<<setw(15)<<Rs[i*9+2]<<endl;
		cout<<Rs[i*9+3]<<setw(15)<<Rs[i*9+4]<<setw(15)<<Rs[i*9+5]<<endl;
		cout<<Rs[i*9+6]<<setw(15)<<Rs[i*9+7]<<setw(15)<<Rs[i*9+8]<<endl;
	} */
	
	/* for(int i=4086;i<4095;i++)
		cout<<Rs[i]<<endl; */
	// Free device memory
	cudaFree(dev_matchfeature_NO);
	cudaFree(dev_map_feature_NO);
	cudaFree(dev_matchfeature_info);
	cudaFree(dev_map_feature_info);
	cudaFree(dev_WL);
	cudaFree(dev_Rs);
	cudaFree(dev_parm);
	cudaFree(dev_P3P_NO);
	cudaFree(dev_ransac_match_num);
	
}

void launchVOKernel_dub(std::vector<keepfeature> matchfeature , std::vector<keepfeature> map_feature,std::vector<Ransac_Pos> &ransac_match,double *GPU_WL,double *GPU_Rs,int *ransac_match_num,double l_u0,double l_v0,double l_fu,double l_fv)
{
	int thredperblock = NT;
	int sol_num=(matchfeature.size()*(matchfeature.size()-1)*(matchfeature.size()-2))/6; //n取3
	int blocknum=(sol_num+thredperblock-1)/thredperblock;
	/////////////////// allocate memory on the cpu side//////////////////////
	int *matchfeature_NO,*map_feature_NO;//編號
	double *matchfeature_info, *map_feature_info;
	int *P3P_NO;//建model的特徵點編號
	//float *WL, *Rs;
	double *parm;
	//int *ransac_match_num;//符合共識集合的數量
	matchfeature_NO = (int*)malloc(matchfeature.size()* sizeof(int));
	map_feature_NO = (int*)malloc(map_feature.size()* sizeof(int));
	matchfeature_info = (double*)malloc(6 *matchfeature.size()* sizeof(double));//hx hy hz ix iy
	map_feature_info = (double*)malloc(6 *map_feature.size()* sizeof(double));//hx hy hz ix iy
	//WL=(float*)malloc(3 *sol_num* sizeof(float));//x y z
	//Rs=(float*)malloc(9 *sol_num* sizeof(float));//11 12 13 21 22 23 31 32 33
	P3P_NO=(int*)malloc(3 *sol_num* sizeof(int));
	parm=(double*)malloc(6 * sizeof(double));//l_u0,l_v0,l_fu,l_fv,map_feature.size()
	//ransac_match_num=(int*)malloc(sol_num* sizeof(int));
	////////////////// put data in the host memory////////////////////////
	int times=0;
	for(int g=0;g<matchfeature.size()-2;g++)
	{
		for(int g_two=g+1 ;  g_two< matchfeature.size()-1 ; g_two++ )
		{
			for(int g_three=g_two+1 ;  g_three< matchfeature.size() ; g_three++ )
			{
				P3P_NO[times]=g;
				P3P_NO[times+1]=g_two;
				P3P_NO[times+2]=g_three;
				times+=3;
			}
		}
	}
	/* for(int i=0;i<3*sol_num;i+=3)
	{
		cout<<P3P_NO[i]<<setw(5)<<P3P_NO[i+1]<<setw(5)<<P3P_NO[i+2]<<endl;
	} */
	parm[0]=l_u0;parm[1]=l_v0;parm[2]=l_fu;parm[3]=l_fv;parm[4]=sol_num;parm[5]=map_feature.size();
	
	for(int i=0;i<matchfeature.size();i++)
	{
		matchfeature_NO[i]=matchfeature[i].num;
		matchfeature_info[i*6]=matchfeature[i].hx;
		matchfeature_info[i*6+1]=matchfeature[i].hy;
		matchfeature_info[i*6+2]=matchfeature[i].hz;
		matchfeature_info[i*6+3]=matchfeature[i].l_ix;
		matchfeature_info[i*6+4]=matchfeature[i].l_iy;
		matchfeature_info[i*6+5]=matchfeature[i].depthvalue;
	}

	for(int j=0;j<map_feature.size();j++)
	{
		map_feature_NO[j]=map_feature[j].num;
		map_feature_info[j*6]=map_feature[j].hx;
		map_feature_info[j*6+1]=map_feature[j].hy;
		map_feature_info[j*6+2]=map_feature[j].hz;
		map_feature_info[j*6+3]=map_feature[j].l_ix;
		map_feature_info[j*6+4]=map_feature[j].l_iy;
		map_feature_info[j*6+5]=map_feature[j].depthvalue;
	}
		
	////////////////// allocate the memory on the GPU ///////////////////////
	
	int *dev_matchfeature_NO,*dev_map_feature_NO;//編號
	double *dev_matchfeature_info, *dev_map_feature_info;
	double *dev_WL,*dev_Rs;
	int *dev_P3P_NO;//建model的特徵點編號
	double *dev_parm;
	int *dev_ransac_match_num;//符合共識集合的數量
	cudaMalloc((void **)&dev_matchfeature_NO, matchfeature.size()* sizeof(int));
	cudaMalloc((void **)&dev_map_feature_NO, map_feature.size()* sizeof(int));
	cudaMalloc((void **)&dev_matchfeature_info, 6 *matchfeature.size()* sizeof(double));
	cudaMalloc((void **)&dev_map_feature_info, 6 *map_feature.size()* sizeof(double));
	cudaMalloc((void **)&dev_WL, 3 *sol_num* sizeof(double));
	cudaMalloc((void **)&dev_Rs, 9 *sol_num* sizeof(double));
	cudaMalloc((void **)&dev_P3P_NO, 3 *sol_num* sizeof(int));
	cudaMalloc((void **)&dev_parm, 6*sizeof(double));
	cudaMalloc((void **)&dev_ransac_match_num, sol_num* sizeof(int));
	////////////////// copy the arrays to the GPU/////////////////////////
	cudaMemcpy(dev_matchfeature_NO, matchfeature_NO, matchfeature.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_map_feature_NO, map_feature_NO, map_feature.size()* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matchfeature_info, matchfeature_info, 6 *matchfeature.size()* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_map_feature_info, map_feature_info, 6 *map_feature.size()* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_WL, GPU_WL, 3 *sol_num* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Rs, GPU_Rs, 9 *sol_num* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_P3P_NO, P3P_NO, 3 *sol_num* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_parm, parm, 6* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ransac_match_num, ransac_match_num, sol_num* sizeof(int), cudaMemcpyHostToDevice);
	
		
	cout<<"called launchVOKernel_dub ok"<<endl;
	cout<<"blocknum"<<setw(10)<<blocknum<<endl;
	cout<<"thredperblock"<<setw(10)<<thredperblock<<endl;
	
	// Get start time event
	cudaEvent_t start_kernel, stop_kernel;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
	cudaEventRecord(start_kernel, 0);
	
	GPU_P3P_dub << < blocknum, thredperblock >> > (dev_matchfeature_NO,dev_map_feature_NO,dev_matchfeature_info,dev_map_feature_info,dev_P3P_NO,dev_parm,dev_WL,dev_Rs,dev_ransac_match_num);

	
	// Get stop time event
	cudaEventRecord(stop_kernel, 0);
	cudaEventSynchronize(stop_kernel);
	// Compute execution time
	float kernelTime;
	cudaEventElapsedTime(&kernelTime, start_kernel, stop_kernel);
	printf("GPU double time: %13f msec\n", kernelTime);
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
	
	cudaMemcpy(GPU_WL, dev_WL, 3 *sol_num* sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(GPU_Rs, dev_Rs, 9 *sol_num* sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ransac_match_num, dev_ransac_match_num, sol_num* sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop_tra, 0);
	cudaEventSynchronize(stop_tra);

	// Compute execution time
	float transferTime;
	cudaEventElapsedTime(&transferTime, start_tra, stop_tra);
	printf("GPU transfer time: %13f msec\n", transferTime);
	cout<<"---------------------------------------------------"<<endl;
	cudaEventDestroy(start_tra);
	cudaEventDestroy(stop_tra);
	
	
		
	/* cout<<"-----------------------GPU_ransac_match--------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		cout<<ransac_match_num[i]<<endl;
	}
	cout<<"-----------------------GPU_ransac_match--------------------------"<<endl;
	cout<<"sol num "<<setw(5)<<sol_num<<endl; */
	
	/* cout<<"-----------------------GPU sol camera pose----------------------------"<<endl; 
	for(int i=0;i<sol_num;i++)
	{
		cout<<GPU_WL[i*3]<<setw(15)<<GPU_WL[i*3+1]<<setw(15)<<GPU_WL[i*3+2]<<endl;
	}
	cout<<"-----------------------GPU sol rotation matrix----------------------------"<<endl;  */
	/* for(int i=0;i<sol_num;i++)
	{
		cout<<Rs[i*9]<<setw(15)<<Rs[i*9+1]<<setw(15)<<Rs[i*9+2]<<endl;
		cout<<Rs[i*9+3]<<setw(15)<<Rs[i*9+4]<<setw(15)<<Rs[i*9+5]<<endl;
		cout<<Rs[i*9+6]<<setw(15)<<Rs[i*9+7]<<setw(15)<<Rs[i*9+8]<<endl;
	} */
	
	/* for(int i=4086;i<4095;i++)
		cout<<Rs[i]<<endl; */
	// Free device memory
	cudaFree(dev_matchfeature_NO);
	cudaFree(dev_map_feature_NO);
	cudaFree(dev_matchfeature_info);
	cudaFree(dev_map_feature_info);
	cudaFree(dev_WL);
	cudaFree(dev_Rs);
	cudaFree(dev_parm);
	cudaFree(dev_P3P_NO);
	cudaFree(dev_ransac_match_num);
	
}


