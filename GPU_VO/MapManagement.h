
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
using namespace std;


class feature {
public:
	int laplacian;
	float descriptor[64] = {0};
	int ix;
	int iy;
	int num;

};

class keepfeature {
public:
	
	double l_ix;
	double l_iy;
	int num;
	double hx;//世界座標
	double hy;
	double hz;
	double depthvalue;
};

#endif

