#ifndef _main_h_
#define _main_h_

#include <cv.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "VO.h"

class SHLiuDlg
{

public:

	VO BinocularVO;

	vector<SingleFeature> image_feature;

    
	double m_SampleTime_c;
	double m_FeatureMapNum_c;
	double m_on_image_num ;
	double m_threshold;
	double m_ConsensusSet_threshold;
	
	//void ShowImage( IplImage *Image, CWnd *Show );
	//void GetImage();
	//void GetDImage();
	//void GetCBDImage();
	void DoEvent();

	//void ShowDImage( IplImage *Image, CWnd *Show );
	//void ShowCBDImage( IplImage *Image, CWnd *Show );

	//void ShowPhotoCount( CWnd *pWnd );
	//void SaveImage(IplImage *Image);
	//void SaveDImage(IplImage *Image);
	//void SaveCBDImage(IplImage *Image);
	//void SaveDepth(unsigned short *depth);
	//void SaveFeatureData();
	//void TestSaveDepth(unsigned short *depth);
	//void DeletePhoto();
	//void DeleteDPhoto();
	//void DeleteCBDPhoto();
	//void DeleteDepth();
	//void DeleteFeatureData();
	void LoadImage();
	void LoadDImage();
	void LoadCBDImage();
	void LoadDepth();
	void LoadFeatureData();
	void LoadDepthData();
	//void PickPhotoCBDImage(IplImage *Image);
	//void PickPhotoDImage(IplImage *Image);
	//void PickPhotoImage(IplImage *Image);
	
	IplImage *pImage;
	IplImage *pDImage;
	IplImage *pCBDImage;
	
	Mat Image;
	
	
	
	//void SURF( const IplImage* Image );
	//void GetFeatureData( const IplImage* image_gray, const CvSURFParams params,const IplImage* pImage );
	void WORK(const Mat l_image, const double SampleTime);
	
	
	int extended; //*************************************************************** 16_2 or 128_1 or 64_0
	double hessian_threshold;
	int octaves;
	int octave_layers;

	
	int PhotoCount;
	int PickPhotoCount;
	//int checkfinish;
	bool in_image0=false;
	bool in_image1=false;
	bool in_image2=false;
	bool in_image3=false;
	bool in_image4=false;
	bool loadcheack=false;

	//double m_SampleTime;
	int m_Frequency;
	int m_Frequency2;
	int m_Map_FeatureNum;
	int m_OnImage_FeatureNum;

	int m_Time1;
	int m_Time2;
	int m_Time3;
	int m_Time4;
	int m_Time5;
	int m_Time6;
	int m_Time7;
	int m_Time8;
	int m_Time9;
	int m_Time10;
	int m_Time11;

	double time1;
	double time2;
	double time3;
	double time4;
	double time5;
	double time6;
	double time7;
	double time8;
	double time9;
	double timecontinum[10];
	double ixx[20];
	double iyy[20];
	int ii;
	int QQ;
	int xx[20];
	int yy[20];
	int ix_pre_new;
	int iy_pre_new;
	
	double scale;
	int show_image_num;
    int show_map_num;
	int show_feature_region;
	int show_Search_Window_Size;
	int show_all_feature;
	int txt;
	double SampleTime;
	
	vector<double> SampleTime_temp;
	
	
	double m_Camerax_c;
	double m_Cameray_c;
	double m_Cameraz_c;
	
	private:
	unsigned short* depth;


};
#endif