
#ifndef _MapManagement_h_
#define _MapManagement_h_

#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <cv.h>

#include "SURF.h"

#define dDscpt 64

using namespace std;

class SURFParams
{
public:
	int hessianThreshold;
	int nOctaves;
	int nOctavelayers;
};
class SingleFeature
{
public:

	double	ix;
	double 	iy;
	int		laplacian; // trace
	int		size; // scale
	float	dir; // 特徵方向
	float	hessian; // 行列式值
	float	descriptor[dDscpt]; // 描述向量
	double	depthvalue;
	
	bool    on_map;
	bool    is_close;
	bool    is_similar;
	
};

class PairFeature
{
public:
	
	int		num;

	double 	ix;
	double 	iy;
	int		laplacian; // trace
	int		size; // scale
	float	dir; // 特徵方向
	float	hessian; // 行列式值
	float	descriptor[dDscpt]; // 描述向量
	double	depthvalue;
	double  pre_depth;

	double 	r_ix;
	double 	r_iy;

	bool    on_map;
	bool    is_close;
	bool    is_similar;
	bool    match;
	bool    new_add;
};

class MapFeature
{
public:

	int		num; // 特徵點地圖編號
	
	double	hx;
	double	hy;
	double	hz;

	double	ix;
	double 	iy;
	double	ix_pre;
	double 	iy_pre;
	double	ix_pre_new;
	double 	iy_pre_new;
	double	depthvalue;
	double  pre_depth;
	int		laplacian; // trace
	int		size; // scale
	float	dir; // 特徵方向
	float	hessian; // 行列式值
	float	original_descriptor[dDscpt]; // 原始的描述向量
	float	current_descriptor[dDscpt]; // 不斷更新的描述向量

	int     search_window_size;

	//double	r_ix;
	//double	r_iy;
	
	bool    on_image;
	bool    new_add;
	bool    old_match;
	bool	at_front;
	bool    is_near;
	int		run_times;
	int		detect_times;
	bool	match_stable;

};

class Keep_Feature
{
public:
	int		num; // 特徵點地圖編號

	double	hx;//世界座標
	double	hy;//世界座標
	double	hz;//世界座標

	double	estimate_hx;//此刻回推世界座標
	double	estimate_hy;//此刻回推世界座標
	double	estimate_hz;//此刻回推世界座標

	double  translation_error[3];//估測與真實的平移誤差向量
	double  translation_error_magnitude; //估測與真實的平移誤差大小
	double  rotation_error_angles[3];//估測與真實的角度誤差向量

	double  r_hx;//前一張keyframe視線向量
	double  r_hy;//前一張keyframe視線向量
	double  r_hz;//前一張keyframe視線向量
	double	l_ix;
	double 	l_iy;
	double	l_ix_pre;
	double 	l_iy_pre;
	double	r_ix;
	double 	r_iy;
	double	depthvalue;
	double  pre_depth;
	int		laplacian; // trace
	int		size; // scale
	float	dir; // 特徵方向
	float	hessian; // 行列式值
	float	original_descriptor[dDscpt]; // 原始的描述向量
	float	current_descriptor[dDscpt]; // 不斷更新的描述向量

	int     search_window_size;
	bool    match;

	int times;  //次數
	int stable; //穩定次數
	//int unstable;
	int count; //檢查次數
	int filter;
	int appear;//出現

	bool	correction_outlier;//修正時是否為outlier
	bool    on_image;
	bool    new_add;
	bool    old_match;
	bool	at_front;
	bool    is_near;
	int		run_times;
	int		detect_times;
	bool	match_stable;
	int		disappear_image_times;
	int		number_times;
	int		outlier_times;
	int		inlier_times;
	int		consensus_time=0;
};

class MapManagement
{
public:
	
	OpevCvSURF surf_OpCV;
	
	int PhotoCount;
	cv::Mat DepthImage;
	
	void Initial_MapManagement(void);	
	void GetFeatureData( const Mat image_gray, const SURFParams params, vector<SingleFeature>& feature);
	void Image_Correction( double& ix, double &iy, const double fu, const double fv, const double u0, const double v0, const double coefficient[8] );
	void Erase_Bad_Feature( vector<Keep_Feature>& map_feature, vector<int>& erase_map_feature , vector<int>& erase_disappear_map_feature, int& on_image_num, const int max_on_image_num, const int erase_times, const double erase_ratio );
	void Add_New_Feature( vector<SingleFeature> feature, vector<SingleFeature> pair_feature, vector<MapFeature>& map_feature, vector<int>& add_map_feature, int& on_image_num, const int max_on_image_num, const int add_window_size, const double similar_threshold, const int extended);


private:

	
	double l_u0;
	double l_fu;
	double r_u0;
	double r_fu;	
	double L;
	double minimum_distance;
	double maximum_distance;

	double abcdefgh[8];

		int num;

};

#endif

