#ifndef _VO_h_
#define _VO_h_

#include <fstream>
#include <iomanip>

#include <cv.h>

#define PI 3.1415926

#include "MapManagement.h"
#include "RANSAC.h"
#include "DataAssociation.h"

using namespace std;

class P3dx_Pos
{
public:
	double x;
	double y;
	double z;
	double x_dir;
	double y_dir;
	double z_dir;
};
class feature_vec
{
public:
	double estimate[3] = {0};
	double current[3] = {0};
};
class Correction_error
{
public:
	double translation_error[3];
	double rotation_error_angle[3];
};
class feature_Pos//攝影機坐標系下特徵點座標
{
public:
	double c_x;
	double c_y;
	double c_z;
	double pre_c_x;
	double pre_c_y;
	double pre_c_z;
};
class Predict_Pos
{
public:
	double m[3];//當下特徵點重心
	double s[3];//前一刻特徵點重心
	double X[3];
	double Rs[3][3];

};

class VO
{
public:

	RANSAC_ho link_RANSAC;
	
	vector<Keep_Feature> map_feature;
	vector<Keep_Feature> last_map_feature;
	vector<Keep_Feature> matchfeature;
	vector<Keep_Feature> Gpu_mapfeature;
	vector<feature_Pos> feature;//用於預測的特徵點資訊
	vector<Ransac_Pos> VO_Pos;
	vector<Ransac_Pos> Correction_error_Pos;//攝影機位置修正量


	int Number_Detection;
	int num_two;
	int on_image_num;
	int hessian_threshold;
	int location[3];

	int pre_new_add;//前一刻新增特徵點數量
	int ray; //keyframe間穩定特徵點視線向量差值符合門檻值個數
	int match_stable_num;//當下穩定特徵點數量
	
	void Initial_VO();
	void Run_VO( const Mat l_image, const double sample_time,int PhotoCount);//,bool isrunning,bool isturnning);

	int DataAssociation_Time;
	int Erase_Bad_Feature_Time;
	int Add_New_Feature_Time;
	int Match_Time;
	int ransac_time;
	
	double C_A;
	double tt;
	int numm;

	bool found;
	bool lost;
	bool firstmove;
	bool firststop;
	int hessian;
	int PhotoCount;
	int feature_num;
	int stoptime;
	int match_number ;
private:
	
	int extended;
	int octaves;
	int octave_layers;
	int hessian_error;
	int minimum_hessian_threshold;
	int maximum_hessian_threshold;
	int minimum_Search_Window_Size;
	int maximum_Search_Window_Size;
	int add_window_size;
	double bino_d_match_threshold;
	double original_d_match_threshold;
	double current_d_match_threshold;
	double absolute_d_match_threshold;
	double similar_threshold;
	int erase_times;
	double erase_ratio;
	int max_on_image_num;
	int Map_Covariance_Predict;
	double maximum_distance;
	double Initial_State_Covariance;
	double Sigma_wv;
	double Sigma_ww;
	double Sigma_v;


	double l_u0;
	double l_v0;
	double l_fu;
	double l_fv;
	double l_coefficient[8];
	double r_u0;
	double r_v0;
	double r_fu;
	double r_fv;
	double r_coefficient[8];	
	double L;
	
	MapManagement map;

	DataAssociation datamatch;

	int count;
	int hessian_threshold_add;
	int hessian_threshold_subtract;


	void Find_Feature(const Mat l_image , vector<SingleFeature>& l_feature , vector<int>& find_map_feature);
	void Save_Feature( vector<PairFeature>&pair_feature , vector<int>& add_map_feature);
	void Comparison_Feature_original_descriptor( vector<PairFeature> &pair_feature );//,bool isMoveDone,bool isHeadingDone);
	void Comparison_Feature_last_map_feature_descriptor( vector<PairFeature> &pair_feature );
	void Comparison_Feature( vector<SingleFeature>&l_feature , vector<PairFeature>&pair_feature );//, bool isMoveDone,bool isHeadingDone);
	void Matching_CPU(vector<Keep_Feature> &map_feature, vector<PairFeature> &pair_feature,vector<int>&cpu_matcher);
	void MatchUpdate_FromGPU_Result(vector<PairFeature> &pair_feature,vector<int>&gpu_matcher);
	
	void Keyfeature_Selection();
	void Predict_Position_State();
	void Correction();
	void Correction_Rotation(vector<Ransac_Pos> &Correction_error, feature_vec pos, int inlier_num);
	void Calculate_Landmark_Current();
	void SearchOutlier();
	void Run_PeopleDetect_slam(const IplImage* l_image_gray);
	void MatixMul(double **M, double **S, int num, double R[3][3]);
	void VO_Set_Hessian_Threshold(int& hessian_threshold, const int on_image_num);
	void MatrixCopy( CvMat *mat, CvMat *submat, const int x, const int y, const int subx, const int suby, const int width, const int height );
};

#endif