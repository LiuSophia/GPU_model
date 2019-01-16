
#ifndef _RANSAC_h_
#define _RANSAC_h_

#include "P3P.h"
#include "MapManagement.h"

#define PI 3.1415926

class Ransac_Pos
{
public:
	vector<int> match_set;
	double X[6];
	double Rs[3][3];
};

class RANSAC_ho
{
public:

	vector<Ransac_Pos> average_ransac_match;
	vector<Ransac_Pos> ransac_match;
	vector<Keep_Feature> matchfeature;
	vector<Keep_Feature> map_feature;
	//vector<Keep_Feature> last_map_feature;
	P3PSolver p3p_link;

	void Run_RANSAC( vector<Keep_Feature> matchfeature , vector<Keep_Feature> map_feature,int PhotoCount, double l_u0,double l_v0,double l_fu,double l_fv);
	int match_number ;
	int ransac_time;
	int map_num_threshold = 0;
private:

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

	//int PhotoCount; 

	
//	void Initial_RANSAC(void);

	

	CvMat *W;
	CvMat *I;
};


#endif