#include "P3P.h"




void P3PSolver::Norm_shen(CvMat* A, CvMat *B)
{
	double number_norm = cvNorm(A, 0, CV_DIFF_L2);
	cvmSet(B, 0, 0, cvmGet(A, 0, 0) / number_norm);
	cvmSet(B, 1, 0, cvmGet(A, 1, 0) / number_norm);
	cvmSet(B, 2, 0, cvmGet(A, 2, 0) / number_norm);
}


void P3PSolver::SetPointsCorrespondance_shen(double *ptw, double *pti)
{
	W = cvCreateMat(3, 3, CV_64FC1);
	I = cvCreateMat(2, 3, CV_64FC1);
	cvSetZero(W);
	cvSetZero(I);
	for (int i = 0; i<3; i++)
	{
		cvmSet(W, 0, i, *(ptw + i * 3 + 0));
		cvmSet(W, 1, i, *(ptw + i * 3 + 1));
		cvmSet(W, 2, i, *(ptw + i * 3 + 2));

		cvmSet(I, 0, i, *(pti + i * 2));
		cvmSet(I, 1, i, *(pti + i * 2 + 1));
	}
}


void P3PSolver::MatrixCopy_shen(CvMat *mat, CvMat *submat, const int row, const int col, const int subrow, const int subcol, const int height, const int width)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cvmSet(submat, subrow + j, subcol + i, cvmGet(mat, row + j, col + i));
		}
	}
}

void P3PSolver::Solve_one_shen(vector<Matrix>& X, vector< Rectangular >& Rs, double aa, double bb, double cc,double l_u0,double l_v0,double l_fu,double l_fv)
{
	//------------------------------------------宣告--------------------------------------------------------------------
	CvMat* W_one = cvCreateMat(3, 1, CV_64FC1);
	CvMat* W_two = cvCreateMat(3, 1, CV_64FC1);
	CvMat* W_three = cvCreateMat(3, 1, CV_64FC1);

	CvMat* VAB = cvCreateMat(3, 1, CV_64FC1);
	CvMat* VAC = cvCreateMat(3, 1, CV_64FC1);
	CvMat* VBC = cvCreateMat(3, 1, CV_64FC1);

	CvMat* CA = cvCreateMat(3, 1, CV_64FC1);
	CvMat* CB = cvCreateMat(3, 1, CV_64FC1);
	CvMat* CC = cvCreateMat(3, 1, CV_64FC1);

	CvMat* WQ = cvCreateMat(3, 1, CV_64FC1);
	CvMat* WP = cvCreateMat(3, 1, CV_64FC1);

	CvMat* P1 = cvCreateMat(4, 1, CV_64FC1);
	CvMat* P2 = cvCreateMat(4, 1, CV_64FC1);
	CvMat* P3 = cvCreateMat(4, 1, CV_64FC1);

	CvMat* NP1 = cvCreateMat(3, 1, CV_64FC1);
	CvMat* NP2 = cvCreateMat(3, 1, CV_64FC1);
	CvMat* NP3 = cvCreateMat(3, 1, CV_64FC1);

	CvMat* VCX = cvCreateMat(3, 1, CV_64FC1);
	CvMat* VCY = cvCreateMat(3, 1, CV_64FC1);
	CvMat* VCZ = cvCreateMat(3, 1, CV_64FC1);

	CvMat* Vla = cvCreateMat(3, 1, CV_64FC1);
	CvMat* Vlb = cvCreateMat(3, 1, CV_64FC1);
	CvMat* Vlc = cvCreateMat(3, 1, CV_64FC1);

	CvMat* WA1 = cvCreateMat(3, 1, CV_64FC1);
	CvMat* WB1 = cvCreateMat(3, 1, CV_64FC1);
	CvMat* WC1 = cvCreateMat(3, 1, CV_64FC1);

	CvMat* vcx = cvCreateMat(3, 1, CV_64FC1);
	CvMat* vcy = cvCreateMat(3, 1, CV_64FC1);
	CvMat* vcz = cvCreateMat(3, 1, CV_64FC1);

	CvMat* BIGA = cvCreateMat(3, 4, CV_64FC1);
	CvMat* BIGA_W = cvCreateMat(3, 4, CV_64FC1);
	CvMat* BIGA_U = cvCreateMat(3, 3, CV_64FC1);
	CvMat* BIGA_V = cvCreateMat(4, 4, CV_64FC1);

	CvMat* WR = cvCreateMat(3, 1, CV_64FC1);
	CvMat* WL = cvCreateMat(3, 1, CV_64FC1);

	CvMat* R1 = cvCreateMat(3, 3, CV_64FC1);
	CvMat* R2 = cvCreateMat(3, 3, CV_64FC1);
	CvMat* R3 = cvCreateMat(3, 3, CV_64FC1);

	CvMat* Rext = cvCreateMat(3, 3, CV_64FC1);
	CvMat* text = cvCreateMat(3, 1, CV_64FC1);

	//------------------------------------------------------------------------------------------------------------------
	
	//l_u0=para[0];l_v0=para[1];l_fu=para[2];l_fv=para[3];
	//copy xyz form world frame
	MatrixCopy_shen(W, W_one, 0, 0, 0, 0, 3, 1);
	MatrixCopy_shen(W, W_two, 0, 1, 0, 0, 3, 1);
	MatrixCopy_shen(W, W_three, 0, 2, 0, 0, 3, 1);


	//Compute Rab Rac Rbc in world frame
	cvSub(W_two, W_one, VAB);
	cvSub(W_three, W_one, VAC);
	cvSub(W_three, W_two, VBC);

	//Length of edge between control points
	double Rab = cvNorm(VAB, 0, CV_DIFF_L2);		//cvNorm(輸入CvMat資料結構,輸入CvMat資料結構或為空,參數或代號,輸入CvMat遮罩)
	double Rac = cvNorm(VAC, 0, CV_DIFF_L2);
	double Rbc = cvNorm(VBC, 0, CV_DIFF_L2);

	cvReleaseMat(&VBC); //release Vbc not useful

	//Get norm of each ray in camera frame
	//CA0=[(IA(1,1)-u0)/fu;(IA(2,1)-v0)/fv;1];
	//CB0=[(IB(1,1)-u0)/fu;(IB(2,1)-v0)/fv;1];
	//CC0=[(IC(1,1)-u0)/fu;(IC(2,1)-v0)/fv;1];
	cvmSet(CA, 0, 0, (cvmGet(I, 0, 0) - l_u0) / l_fu);
	cvmSet(CA, 1, 0, (cvmGet(I, 1, 0) - l_v0) / l_fv);
	cvmSet(CA, 2, 0, 1);

	cvmSet(CB, 0, 0, (cvmGet(I, 0, 1) - l_u0) / l_fu);
	cvmSet(CB, 1, 0, (cvmGet(I, 1, 1) - l_v0) / l_fv);
	cvmSet(CB, 2, 0, 1);

	cvmSet(CC, 0, 0, (cvmGet(I, 0, 2) - l_u0) / l_fu);
	cvmSet(CC, 1, 0, (cvmGet(I, 1, 2) - l_v0) / l_fv);
	cvmSet(CC, 2, 0, 1);

	//Normalize
	Norm_shen(CA, CA);
	Norm_shen(CB, CB);
	Norm_shen(CC, CC);
	
	
	double Rab1 = cvNorm(CB, CA, CV_DIFF_L2);//向量長度計算 unit mm//歐基里德距離
	double Rac1 = cvNorm(CC, CA, CV_DIFF_L2);
	double Rbc1 = cvNorm(CC, CB, CV_DIFF_L2);


	//Cosine of angles
	double Calb, Calc, Cblc;

	//Compute Calb Calc Cblc using Law of Cosine
	Calb = (2 - Rab1*Rab1) / 2;
	Calc = (2 - Rac1*Rac1) / 2;
	Cblc = (2 - Rbc1*Rbc1) / 2;

	//use hx hy hz to replace abc
	double a = aa;
	double b = bb;
	double c = cc;

	//%Get cosine of the angles
	//Clab=(a^2+Rab^2-b^2)/(2*a*Rab);
	//Clac=(a^2+Rac^2-c^2)/(2*a*Rac);
	double Clab = (a*a + Rab*Rab - b*b) / (2 * a*Rab);
	double Clac = (a*a + Rac*Rac - c*c) / (2 * a*Rac);

	//%Get scale along norm vector
	//Raq=a*Clab;
	//Rap=a*Clac;
	double Raq = a*Clab;
	double Rap = a*Clac;

	/* printf("CPU a  %.20f\n",a);
	printf("CPU b  %.20f\n",b); */
	
	//%Get norm vector of plane P1 P2
	//VAC=WC-WA;
	//VAB=WB-WA;
	double VAB_norm = cvNorm(VAB, 0, CV_DIFF_L2);
	double VAC_norm = cvNorm(VAC, 0, CV_DIFF_L2);

	for (int k = 0; k<3; k++)
	{
		cvmSet(WQ, k, 0, cvmGet(W_one, k, 0) + Raq*cvmGet(VAB, k, 0) / VAB_norm);
		cvmSet(WP, k, 0, cvmGet(W_one, k, 0) + Rap*cvmGet(VAC, k, 0) / VAC_norm);
	}

	double DP1, DP2, DP3;

	//%Compute Plane P1 P2 P3

	//NP1=VAB/norm(VAB);
	//DP1=-NP1'*WQ;
	//P1=[NP1;DP1];
	Norm_shen(VAB, NP1);
	DP1 = cvDotProduct(NP1, WQ);
	MatrixCopy_shen(NP1, P1, 0, 0, 0, 0, 3, 1);
	cvmSet(P1, 3, 0, -DP1);

	//NP2=VAC/norm(VAC);
	//DP2=-NP2'*WP;
	//P2=[NP2;DP2];
	Norm_shen(VAC, NP2);
	DP2 = cvDotProduct(NP2, WP);
	MatrixCopy_shen(NP2, P2, 0, 0, 0, 0, 3, 1);
	cvmSet(P2, 3, 0, -DP2);

	cvSub(CB, CA, VCX);
	cvSub(CC, CA, VCY);

	if ( (cvmGet(VCX, 0, 0) * cvmGet(VCY, 1, 0) - cvmGet(VCY, 0, 0) * cvmGet(VCX, 1, 0)) > 0 )	//計算視線向量外積Z方向的分量判斷使用何種排列(負為AC X AB)
	{
		cvCrossProduct(VAC, VAB, NP3);
	}
	else
	{
		cvCrossProduct(VAB, VAC, NP3);
	}
	
	/*
	if(cvmGet(NP3,2,0)>0)
	{
		cvmSet(NP3, 0, 0, -cvmGet(NP3, 0, 0));
		cvmSet(NP3, 1, 0, -cvmGet(NP3, 1, 0));
		cvmSet(NP3, 2, 0, -cvmGet(NP3, 2, 0));
	}
	*/

	double NP3_norm = cvNorm(NP3, 0, CV_DIFF_L2);

	//if (NP3_norm == 0) break;// 給定 a b c 不會有0問題

	Norm_shen(NP3, NP3);
	DP3 = cvDotProduct(NP3, W_one);
	MatrixCopy_shen(NP3, P3, 0, 0, 0, 0, 3, 1);
	cvmSet(P3, 3, 0, -DP3);
	
	//BIGA=[P1';P2';P3'];
	//MatrixCopy_shen(P1, BIGA, 0, 0, 0, 0, 4, 1);
	//MatrixCopy_shen(P2, BIGA, 0, 0, 0, 1, 4, 1);
	//MatrixCopy_shen(P3, BIGA, 0, 0, 0, 2, 4, 1);
	for (char k = 0; k < 4;k++)
		cvmSet(BIGA, 0, k, cvmGet(P1, k, 0));
	for (char k = 0; k < 4; k++)
		cvmSet(BIGA, 1, k, cvmGet(P2, k, 0));
	for (char k = 0; k < 4; k++)
		cvmSet(BIGA, 2, k, cvmGet(P3, k, 0));

	//singular value decomposition
	cvSVD(BIGA, BIGA_W, BIGA_U, BIGA_V);

	cvmSet(WR, 0, 0, cvmGet(BIGA_V, 0, 3) / cvmGet(BIGA_V, 3, 3));
	cvmSet(WR, 1, 0, cvmGet(BIGA_V, 1, 3) / cvmGet(BIGA_V, 3, 3));
	cvmSet(WR, 2, 0, cvmGet(BIGA_V, 2, 3) / cvmGet(BIGA_V, 3, 3));

	//%Get length of LR
	//Rar=norm(WA-WR);
	//Rlr=sqrt(a^2-Rar^2);
	double Rar, Rlr;
	Rar = cvNorm(W_one, WR, CV_DIFF_L2);
	//if ((a*a - Rar*Rar) < 0)	continue;
	if(a*a - Rar*Rar>0)
		Rlr = sqrt(a*a - Rar*Rar);//當aa*aa - Rar*Rar<0時，Rlr則為nan
	else
		Rlr = 10e6;
	//Rlr = sqrt(a*a - Rar*Rar);

	//%Get Position of L in world frame
	//WL=WR+NP3*Rlr;
	cvmSet(WL, 0, 0, cvmGet(WR, 0, 0) + cvmGet(NP3, 0, 0)*Rlr);
	cvmSet(WL, 1, 0, cvmGet(WR, 1, 0) + cvmGet(NP3, 1, 0)*Rlr);
	cvmSet(WL, 2, 0, cvmGet(WR, 2, 0) + cvmGet(NP3, 2, 0)*Rlr);
	//cout<< cvmGet(WL, 0, 0)<<setw(10)<<cvmGet(WL, 1, 0)<<setw(10)<<cvmGet(WL, 2, 0)<<endl;
	
	//test //克拉馬公式求三平面解 P1 P2 P3//a1x+b1y+c1z-d1=0 
	//double delta, delta_x, delta_y, delta_z;
	//delta=cvmGet(P1, 0, 0)*cvmGet(P2, 1, 0)*cvmGet(P3, 2, 0)+cvmGet(P2, 0, 0)*cvmGet(P3, 1, 0)*cvmGet(P1, 2, 0)+cvmGet(P3, 0, 0)*cvmGet(P1, 1, 0)*cvmGet(P2, 2, 0)-cvmGet(P3, 0, 0)*cvmGet(P2, 1, 0)*cvmGet(P1, 2, 0)-cvmGet(P1, 0, 0)*cvmGet(P3, 1, 0)*cvmGet(P2, 2, 0)-cvmGet(P2, 0, 0)*cvmGet(P1, 1, 0)*cvmGet(P3, 2, 0);
	//delta_x=cvmGet(P2, 1, 0)*cvmGet(P3, 2, 0)*cvmGet(P1, 3, 0)+cvmGet(P1, 1, 0)*cvmGet(P2, 2, 0)*cvmGet(P3, 3, 0)+cvmGet(P3, 1, 0)*cvmGet(P2, 3, 0)*cvmGet(P1, 2, 0)-cvmGet(P2, 1, 0)*cvmGet(P1, 2, 0)*cvmGet(P3, 3, 0)-cvmGet(P1, 1, 0)*cvmGet(P2, 3, 0)*cvmGet(P3, 2, 0)-cvmGet(P3, 1, 0)*cvmGet(P1, 3, 0)*cvmGet(P2, 2, 0);
	//delta_y=cvmGet(P1, 0, 0)*cvmGet(P3, 2, 0)*cvmGet(P2, 3, 0)+cvmGet(P2, 0, 0)*cvmGet(P1, 2, 0)*cvmGet(P3, 3, 0)+cvmGet(P3, 0, 0)*cvmGet(P2, 2, 0)*cvmGet(P1, 3, 0)-cvmGet(P3, 0, 0)*cvmGet(P1, 2, 0)*cvmGet(P2, 3, 0)-cvmGet(P1, 0, 0)*cvmGet(P2, 2, 0)*cvmGet(P3, 3, 0)-cvmGet(P2, 0, 0)*cvmGet(P3, 2, 0)*cvmGet(P1, 3, 0);
	//delta_z=cvmGet(P1, 0, 0)*cvmGet(P2, 1, 0)*cvmGet(P3, 3, 0)+cvmGet(P2, 0, 0)*cvmGet(P3, 1, 0)*cvmGet(P1, 3, 0)+cvmGet(P3, 0, 0)*cvmGet(P1, 1, 0)*cvmGet(P2, 3, 0)-cvmGet(P3, 0, 0)*cvmGet(P2, 1, 0)*cvmGet(P1, 3, 0)-cvmGet(P1, 0, 0)*cvmGet(P3, 1, 0)*cvmGet(P2, 3, 0)-cvmGet(P2, 0, 0)*cvmGet(P1, 1, 0)*cvmGet(P3, 3, 0);
	
	//double WR_test[3],WL_test[3];
	//if(delta!=0)
	//{
		//上面係數移項，因此差一個負號
	//	WR_test[0]=-delta_x/delta;	WR_test[1]=-delta_y/delta;	WR_test[2]=-delta_z/delta;
	//}
	//else
	//{
	//	WR_test[0]=1000000;	WR_test[1]=1000000;	WR_test[2]=1000000;
	//}
	
	//WL_test[0]=WR_test[0] + cvmGet(NP3, 0, 0)*Rlr;
	//WL_test[1]=WR_test[1] + cvmGet(NP3, 1, 0)*Rlr;
	//WL_test[2]=WR_test[2] + cvmGet(NP3, 2, 0)*Rlr;
	//cout<<WL_test[0]<<setw(10)<<WL_test[1]<<setw(10)<<WL_test[2]<<endl;
	
	//if(sqrt((cvmGet(WL, 0, 0)-WL_test[0])*(cvmGet(WL, 0, 0)-WL_test[0]))>0.00001)
	//{
	//	cout<<"WL_test"<<endl;
	//	cout<<WL_test[0]<<setw(10)<<WL_test[1]<<setw(10)<<WL_test[2]<<endl;
	//	cout<<"WL"<<endl;
	//	cout<< cvmGet(WL, 0, 0)<<setw(10)<<cvmGet(WL, 1, 0)<<setw(10)<<cvmGet(WL, 2, 0)<<endl;
	//}
	//else
	//{
		//cout<<"pass"<<endl;
	//}
	
	//%Get unprojection ray of image points
	//CA0=[(IA(1,1)-u0)/fu;(IA(2,1)-v0)/fv;1];
	//CB0=[(IB(1,1)-u0)/fu;(IB(2,1)-v0)/fv;1];
	//CC0=[(IC(1,1)-u0)/fu;(IC(2,1)-v0)/fv;1];
	//VCx=CB1-CA1;		%此在前面計算，為了決定AB和AC外積排列%
	//VCy=CC1-CA1;
	//VCz=crossproduct3d(VCx,VCy)';
	//VCy=crossproduct3d(VCz,VCx)';
	
	cvCrossProduct(VCX, VCY, VCZ);
	cvCrossProduct(VCZ, VCX, VCY);

	Norm_shen(VCX, VCX);
	Norm_shen(VCY, VCY);
	Norm_shen(VCZ, VCZ);


	//%Get ray in the world frame
	//Vla=WA-WL;
	//Vlb=WB-WL;
	//Vlc=WC-WL;
	cvSub(W_one, WL, Vla);
	cvSub(W_two, WL, Vlb);
	cvSub(W_three, WL, Vlc);
	Norm_shen(Vla, Vla);
	Norm_shen(Vlb, Vlb);
	Norm_shen(Vlc, Vlc);

	//WA1=WL+1*Vla;
	//WB1=WL+1*Vlb;
	//WC1=WL+1*Vlc;
	cvAdd(WL, Vla, WA1);
	cvAdd(WL, Vlb, WB1);
	cvAdd(WL, Vlc, WC1);

	//vcx=WB1-WA1;
	//vcy=WC1-WA1;
	//vcz=crossproduct3d(vcx,vcy)';
	//vcy=crossproduct3d(vcz,vcx)';
	cvSub(WB1, WA1, vcx);
	cvSub(WC1, WA1, vcy);
	
		
	cvCrossProduct(vcx, vcy, vcz);
	cvCrossProduct(vcz, vcx, vcy);
	
		
	Norm_shen(vcx, vcx);
	Norm_shen(vcy, vcy);
	Norm_shen(vcz, vcz);

	
	
	//R=[VCx VCy VCz]*inv([vcx vcy vcz]);
	MatrixCopy_shen(VCX, R1, 0, 0, 0, 0, 3, 1);
	MatrixCopy_shen(VCY, R1, 0, 0, 0, 1, 3, 1);
	MatrixCopy_shen(VCZ, R1, 0, 0, 0, 2, 3, 1);

	MatrixCopy_shen(vcx, R2, 0, 0, 0, 0, 3, 1);
	MatrixCopy_shen(vcy, R2, 0, 0, 0, 1, 3, 1);
	MatrixCopy_shen(vcz, R2, 0, 0, 0, 2, 3, 1);

	cvTranspose(R2, R2);
	
	
	
	cvMatMul(R1, R2, Rext);//矩陣相乘
	
	for (int k = 0; k< 3; k++)
	{
		cvmSet(text, k, 0, -cvmGet(WL, k, 0));
	}
	cvMatMul(Rext, text, text);

	Matrix temp;
	memset(&temp, 0, sizeof(Matrix));

	temp.data_[0] = cvmGet(WL, 0, 0);
	temp.data_[1] = cvmGet(WL, 1, 0);
	temp.data_[2] = cvmGet(WL, 2, 0);

	X.push_back(temp);

	Rectangular temp2;
	memset(&temp2, 0, sizeof(Rectangular));
	
	for (int k = 0; k<3; k++)
	{
		temp2.data_[k][0] = cvmGet(Rext, k, 0);
		temp2.data_[k][1] = cvmGet(Rext, k, 1);
		temp2.data_[k][2] = cvmGet(Rext, k, 2);
	}

	Rs.push_back(temp2);
	
	/* cout<< cvmGet(Rext, 0, 0)<<setw(10)<<cvmGet(Rext, 0, 1)<<setw(10)<<cvmGet(Rext, 0, 2)<<endl;
	cout<< cvmGet(Rext, 1, 0)<<setw(10)<<cvmGet(Rext, 1, 1)<<setw(10)<<cvmGet(Rext, 1, 2)<<endl;
	cout<< cvmGet(Rext, 2, 0)<<setw(10)<<cvmGet(Rext, 2, 1)<<setw(10)<<cvmGet(Rext, 2, 2)<<endl; */
	
	cvReleaseMat(&W_one);
	cvReleaseMat(&W_two);
	cvReleaseMat(&W_three);

	cvReleaseMat(&VAB);
	cvReleaseMat(&VAC);
	cvReleaseMat(&VBC);

	cvReleaseMat(&CA);
	cvReleaseMat(&CB);
	cvReleaseMat(&CC);
	
	cvReleaseMat(&WQ);
	cvReleaseMat(&WP);

	cvReleaseMat(&P1);
	cvReleaseMat(&P2);
	cvReleaseMat(&P3);

	cvReleaseMat(&NP1);
	cvReleaseMat(&NP2);
	cvReleaseMat(&NP3);

	cvReleaseMat(&VCX);
	cvReleaseMat(&VCY);
	cvReleaseMat(&VCZ);

	cvReleaseMat(&Vla);
	cvReleaseMat(&Vlb);
	cvReleaseMat(&Vlc);

	cvReleaseMat(&WA1);
	cvReleaseMat(&WB1);
	cvReleaseMat(&WC1);

	cvReleaseMat(&vcx);
	cvReleaseMat(&vcy);
	cvReleaseMat(&vcz);

	cvReleaseMat(&BIGA);
	cvReleaseMat(&BIGA_W);
	cvReleaseMat(&BIGA_U);
	cvReleaseMat(&BIGA_V);
	
	cvReleaseMat(&WR);
	cvReleaseMat(&WL);
	
	cvReleaseMat(&R1);
	cvReleaseMat(&R2);
	cvReleaseMat(&R3);
	
	cvReleaseMat(&Rext);
	
	cvReleaseMat(&text);
	
	cvReleaseMat(&W);
	cvReleaseMat(&I);
	
}
