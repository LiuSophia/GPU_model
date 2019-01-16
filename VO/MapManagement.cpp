#include "MapManagement.h"

void MapManagement::Initial_MapManagement()
{

	num = 0;
	PhotoCount = 0;


	char str[50];
	double value;

	fstream in_Camera( "Paratxt//Camera.txt", ios::in );
	if(!in_Camera)	exit(1);
	while( in_Camera >> str >> value )
	{
		if      ( !strcmp( str, "l_u0" ) )	 l_u0 = value;
		else if	( !strcmp( str, "l_fu" ) )   l_fu = value;
		else if ( !strcmp( str, "r_u0" ) )	 r_u0 = value;
		else if ( !strcmp( str, "r_fu" ) )   r_fu = value;
		else if ( !strcmp( str, "L" ) )      L    = value;
	}
	in_Camera.close();



	fstream in_Parameter( "Paratxt//Parameter.txt", ios::in );
	if(!in_Parameter)	exit(1);
	while( in_Parameter >> str >> value )
	{
		if	    ( !strcmp( str, "minimum_distance" ) )	minimum_distance = value;
		else if ( !strcmp( str, "maximum_distance" ) )	maximum_distance = value;

	}
	in_Parameter.close();
}

void MapManagement::GetFeatureData( const Mat image_gray, const SURFParams params, vector<SingleFeature>& feature)
{
	vector< KeyPoint> keypoints_GPU,keypoints_CPU;
	vector< float> descriptors_GPU;
	Mat descriptors_CPU;
	
	//Read DepthImage
	char path[100];
	snprintf(path,sizeof(path), "fig//depth//%d.png", PhotoCount);
	ifstream in_file(path, ios::in);
	if (in_file)
	{
		DepthImage=cv::imread(path,CV_16UC1);
	}
	in_file.close();
	
	
	//SURF with CPU
	surf_OpCV.processSurfWithCpu(image_gray,params.hessianThreshold,keypoints_CPU,descriptors_CPU);
	
	//SURF with GPU
	//surf_OpCV.processSurfWithGpu(image_gray,params.hessianThreshold,keypoints_GPU,descriptors_GPU);
	/* cout<<keypoints_CPU.size()<<endl;
	cout<<descriptors_CPU.rows*descriptors_CPU.cols<<endl; */

	SingleFeature temp;
	for(int i=0;i<keypoints_CPU.size();i++)
	{
		memset( &temp, 0, sizeof(SingleFeature) );
		temp.ix=keypoints_CPU[i].pt.x;
		temp.iy=keypoints_CPU[i].pt.y;
		temp.laplacian=keypoints_CPU[i].class_id;
		temp.size=keypoints_CPU[i].size;
		temp.dir=keypoints_CPU[i].angle;
		temp.hessian=keypoints_CPU[i].response;
		temp.depthvalue = DepthImage.at<unsigned short>(temp.iy, temp.ix);
		
		if(temp.depthvalue<1000 || temp.depthvalue>5000) continue;
		
		float *DCdata=descriptors_CPU.ptr<float>(i);
		for( int j=0 ; j < dDscpt ; j++ )
		{
			
			temp.descriptor[j] = DCdata[j];
		}
		feature.push_back(temp);//丟回vector<SingleFeature>最後
	}
	
	
	/* SingleFeature temp;
	for(int i=0;i<keypoints_GPU.size();i++)
	{
		memset( &temp, 0, sizeof(SingleFeature) );
		temp.ix=keypoints_GPU[i].pt.x;
		temp.iy=keypoints_GPU[i].pt.y;
		temp.laplacian=keypoints_GPU[i].class_id;
		temp.size=keypoints_GPU[i].size;
		temp.dir=keypoints_GPU[i].angle;
		temp.hessian=keypoints_GPU[i].response;
		temp.depthvalue = DepthImage.at<unsigned short>(temp.iy, temp.ix);
		
		if(temp.depthvalue<1000 || temp.depthvalue>5000) continue;
		for( int j=0 ; j < dDscpt ; j++ )
		{
			temp.descriptor[j] = descriptors_GPU[i*dDscpt+j];
		}
		feature.push_back(temp);//丟回vector<SingleFeature>最後
	} */
	
	PhotoCount++;
}

void MapManagement::Erase_Bad_Feature( vector<Keep_Feature>& map_feature, vector<int>& erase_map_feature , vector<int>& erase_disappear_map_feature , int& on_image_num, const int max_on_image_num, const int erase_times, const double erase_ratio )
{


	int match_stable_on_image_num = 0;//比對成功數量初始


	for( int i=0 ; i < (int)map_feature.size() ; i++ )
	{
		if (map_feature[i].match_stable) continue;
		map_feature[i].run_times++;//比對的次數

		if(map_feature[i].on_image)
		{
			map_feature[i].detect_times++;//如果出現在地圖上則增加detect_times
		}

		if( !map_feature[i].match_stable  &&  map_feature[i].run_times <= erase_times  &&  map_feature[i].detect_times >= cvRound(erase_times*erase_ratio) )//如果 map_feature[i].match_stable為否代表還未成為穩定地標，run_times <= erase_times代表還未到達10次的比對次數，detect_times >= cvRound(erase_times*erase_ratio)代表出現在地圖上的次數是否大於7成
		{
			map_feature[i].match_stable = true; // 特徵點比對率符合

			map_feature[i].num = num;
			num++;
		}

		if( map_feature[i].on_image  &&  map_feature[i].match_stable )//在地圖且比對為成功者
		{
			match_stable_on_image_num++;//比對成功數量增加
		}

	}

	for (int i = 0; i < (int)map_feature.size(); i++)			//不存地標
	{
		map_feature[i].number_times++;
		if (!map_feature[i].on_image && map_feature[i].match_stable)
			map_feature[i].disappear_image_times++;

		if(map_feature[i].number_times == 40 && map_feature[i].disappear_image_times>=35)
			erase_disappear_map_feature.push_back(i); // 要刪除的地圖特徵點編號

		if (map_feature[i].number_times == 40)
		{
			map_feature[i].number_times = 0;
			map_feature[i].disappear_image_times = 0;
		}
	}






///////////////////////////////////////////////////////////////////////////////////////////////兩種刪法一種為超過最大地圖量直接刪除比對還不符合的特徵，第二種刪除雖還未超過最大地圖特徵量但已經到達比對次數上限但依然不符合的特徵點
	//if( match_stable_on_image_num >= max_on_image_num )//比對成功數量大於最大地圖數量
	//{
	//	for( int i=0 ; i < (int)map_feature.size() ; i++ )
	//	{
	//		if(!map_feature[i].match_stable)//不為比對成功特徵
	//		{
	//			erase_map_feature.push_back(i); // 要刪除的地圖特徵點編號
	//		}
	//	}
	//}
	//else
	//{
		for( int i=0 ; i < (int)map_feature.size() ; i++ )
		{	
			if( map_feature[i].run_times == erase_times  &&  !map_feature[i].match_stable ) //特徵點比對率低
			{
				if(map_feature[i].inlier_times<5)
				erase_map_feature.push_back(i); // 要刪除的地圖特徵點編號
			}
			/*if(map_feature[i].run_times == erase_times && !map_feature[i].match_stable && map_feature[i].inlier_times<10)
			{
				erase_map_feature.push_back(i); // 要刪除的地圖特徵點編號
			}*/
		}
	//}
	
//////////////////////////////////////////////////////////////////////////////////////////////////


	for( int i=(int)erase_map_feature.size()-1 ; i >= 0 ; i-- ) // 執行刪除
	{
		if(map_feature[erase_map_feature[i]].on_image) // 只能放這裡 
		{
			on_image_num--;
		}

		map_feature.erase( map_feature.begin() + erase_map_feature[i] );//進行刪除動作
	}
	
	for (int i = (int)erase_disappear_map_feature.size() - 1; i >= 0; i--) // 執行刪除
	{

		map_feature.erase(map_feature.begin() + erase_disappear_map_feature[i]);//進行刪除動作
	}
}




void MapManagement::Image_Correction( double& ix, double &iy, const double fu, const double fv, const double u0, const double v0, const double coefficient[8] )
{
	double hx_hz = (ix-u0)/fu;
	double hy_hz = (iy-v0)/fv;

    double r_dist = sqrt(hx_hz*hx_hz+hy_hz*hy_hz);

	double G = 4*coefficient[4]*r_dist*r_dist + 6*coefficient[5]*r_dist*r_dist*r_dist*r_dist + 8*coefficient[6]*hy_hz + 8*coefficient[7]*hx_hz + 1;

	double new_hx_hz =  hx_hz + (hx_hz*(coefficient[0]*r_dist*r_dist + coefficient[1]*r_dist*r_dist*r_dist*r_dist) + 2*coefficient[2]*hx_hz*hy_hz + coefficient[3]*(r_dist*r_dist + 2*hx_hz*hx_hz) )/G;
    double new_hy_hz =  hy_hz + (hy_hz*(coefficient[0]*r_dist*r_dist + coefficient[1]*r_dist*r_dist*r_dist*r_dist) + coefficient[2]*(r_dist*r_dist + 2*hy_hz*hy_hz) + 2*coefficient[3]*hx_hz*hy_hz )/G;
        
	ix = u0 + fu*new_hx_hz;
    iy = v0 + fv*new_hy_hz;
}
