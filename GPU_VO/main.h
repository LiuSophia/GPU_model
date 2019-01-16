#ifndef _main_h_
#define _main_h_


#include <fstream>
#include <iomanip>


//#include "kernelP3P.cuh"
#include <cv.h>

#define PI 3.1415926

#include "MapManagement.h"
#include "RANSAC.h"
using namespace std;


class VO
{
	
public:

	
	RANSAC_ho link_RANSAC;
	void InitialP3P();
	vector<Ransac_Pos> ransac_match;
	vector<keepfeature> match_feature;
	vector<keepfeature> map_feature;
	vector<Ransac_Pos> VO_Pos;
	double l_fu,l_fv,l_u0,l_v0;
	int mapNum,featureNum;
};

#endif