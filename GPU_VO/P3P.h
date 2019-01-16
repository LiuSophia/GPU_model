#ifndef _P3P_sol_
#define _P3P_sol_

#include "MapManagement.h"
class Matrix
{
public:
	  double data_[3];
};

class Rectangular
{
public:
	  double data_[3][3];
};

class P3PSolver
{
	
public:
	void SetPointsCorrespondance_shen(double *ptw, double *pti);
	void Solve_one_shen(vector<Matrix>& X, vector< Rectangular >& Rs, double aa, double bb, double cc,double l_u0,double l_v0,double l_fu,double l_fv);
	
	
	
private:	
	void MatrixCopy_shen(CvMat *mat, CvMat *submat, const int row, const int col, const int subrow, const int subcol, const int width, const int height);
	void Norm_shen(CvMat* A, CvMat *B);
	CvMat *W;
	CvMat *I;
};



#endif