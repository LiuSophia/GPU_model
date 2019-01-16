#include "VO.h"
#include "match.cuh"
#include "kernelVO.cuh"
void VO::Initial_VO()
{
	tt=0;
	numm=0;

	found = false;
	lost = false;
	count = 0;
	hessian_threshold_subtract = 0;
	hessian_threshold_add = 0;

	Ransac_Pos temp;
	for (char w = 0; w < 3; w++)
	{
		for (char h = 0; h < 3; h++)
		{
			temp.Rs[w][h] = 0;
		}
		temp.X[w] = 0;
	}
	temp.Rs[0][0] = 1;
	temp.Rs[2][1] = -1;
	temp.Rs[1][2] = 1;
	VO_Pos.push_back(temp);
	
	char str[50];double value;
	fstream in_Parameter( "Paratxt//Parameter.txt", ios::in );
	if(!in_Parameter)	exit(1);
	while( in_Parameter >> str >> value )
	{
		if	    ( !strcmp( str, "extended" ) )					  extended			    	 = (int)value;
		else if ( !strcmp( str, "hessian_threshold" ) )			  hessian_threshold		     = (int)value;
		else if ( !strcmp( str, "octaves" ) )					  octaves					 = (int)value;
		else if ( !strcmp( str, "octave_layers" ) )				  octave_layers			     = (int)value;
		else if	( !strcmp( str, "hessian_error" ) )				  hessian_error			     = (int)value;
		else if ( !strcmp( str, "minimum_hessian_threshold" ) )	  minimum_hessian_threshold	 = (int)value;
		else if ( !strcmp( str, "maximum_hessian_threshold" ) )	  maximum_hessian_threshold	 = (int)value;
		else if ( !strcmp( str, "minimum_Search_Window_Size" ) )  minimum_Search_Window_Size = (int)value;
		else if ( !strcmp( str, "maximum_Search_Window_Size" ) )  maximum_Search_Window_Size = (int)value;
		else if ( !strcmp( str, "add_window_size" ) )			  add_window_size			 = (int)value;
		else if ( !strcmp( str, "bino_d_match_threshold" ) )	  bino_d_match_threshold	 = value;
		else if ( !strcmp( str, "original_d_match_threshold" ) )  original_d_match_threshold = value;
		else if ( !strcmp( str, "current_d_match_threshold" ) )	  current_d_match_threshold  = value;
		else if ( !strcmp( str, "absolute_d_match_threshold" ) )  absolute_d_match_threshold = value;
		else if ( !strcmp( str, "similar_threshold" ) )			  similar_threshold	         = value;
		else if ( !strcmp( str, "erase_times" ) )			      erase_times			     = (int)value;
		else if ( !strcmp( str, "erase_ratio" ) )			      erase_ratio			     = value;
		else if ( !strcmp( str, "max_on_image_num" ) )	          max_on_image_num			 = (int)value;
		else if ( !strcmp( str, "maximum_distance" ) )            maximum_distance           = value;

	}
	in_Parameter.close();
	
	fstream in_Camera( "Paratxt//Camera.txt", ios::in );
	if(!in_Camera)	exit(1);
	while( in_Camera >> str >> value )
	{ 
		if ( !strcmp( str, "l_u0" ) )							  l_u0						= value;
		else if ( !strcmp( str, "l_v0" ) )						  l_v0						= value;
		else if ( !strcmp( str, "l_fu" ) )                        l_fu						= value;
		else if ( !strcmp( str, "l_fv" ) )                        l_fv						= value;
		else if ( !strcmp( str, "l_coefficient[0]" ) )			  l_coefficient[0]			= value;
		else if ( !strcmp( str, "l_coefficient[1]" ) )			  l_coefficient[1]			= value;
		else if ( !strcmp( str, "l_coefficient[2]" ) )			  l_coefficient[2]			= value;
		else if ( !strcmp( str, "l_coefficient[3]" ) )			  l_coefficient[3]			= value;
		else if ( !strcmp( str, "l_coefficient[4]" ) )			  l_coefficient[4]			= value;
		else if ( !strcmp( str, "l_coefficient[5]" ) )			  l_coefficient[5]			= value;
		else if ( !strcmp( str, "l_coefficient[6]" ) )			  l_coefficient[6]			= value;
		else if ( !strcmp( str, "l_coefficient[7]" ) )			  l_coefficient[7]			= value;
	}
	in_Camera.close();

	PhotoCount=0;
	feature_num=0;
	ransac_time=0;
	
	map.Initial_MapManagement();
	
}

bool Hessian_StrengthVO( const SingleFeature& temp1, const SingleFeature& temp2 ) // 比大小函式(Search_Feature用)
{
	return temp1.hessian > temp2.hessian;
}

void VO::VO_Set_Hessian_Threshold(int& hessian_threshold, const int on_image_num)
{

	if (on_image_num == 0 && hessian_threshold - 1000 >= minimum_hessian_threshold)
	{
		hessian_threshold -= 1000;
	}
	else if (on_image_num < max_on_image_num)
	{
		if (hessian_threshold - 10 * (max_on_image_num - on_image_num) >= minimum_hessian_threshold)
		{
			hessian_threshold -= 10 * (max_on_image_num - on_image_num);
		}

	}
	else if (on_image_num > max_on_image_num)
	{

		if (hessian_threshold + 50 <= maximum_hessian_threshold)
		{
			hessian_threshold += 50;
		}
	}

}


void VO::Calculate_Landmark_Current()
{
	double hx, hy, hz;
	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable)//在地圖且比對為成功者
		{
			//利用當下算出的攝影機位置回推世界座標
			hz = map_feature[i].depthvalue;
			hx = hz*(map_feature[i].l_ix - l_u0) / l_fu;
			hy = hz*(map_feature[i].l_iy - l_v0) / l_fv;

			map_feature[i].estimate_hx= VO_Pos[0].Rs[0][0] * hx + VO_Pos[0].Rs[0][1] * hy + VO_Pos[0].Rs[0][2] * hz + VO_Pos[0].X[0];
			map_feature[i].estimate_hy= VO_Pos[0].Rs[1][0] * hx + VO_Pos[0].Rs[1][1] * hy + VO_Pos[0].Rs[1][2] * hz + VO_Pos[0].X[1];
			map_feature[i].estimate_hz= VO_Pos[0].Rs[2][0] * hx + VO_Pos[0].Rs[2][1] * hy + VO_Pos[0].Rs[2][2] * hz + VO_Pos[0].X[2];

			//當下回推世界座標與原世界座標的平移誤差與大小
			map_feature[i].translation_error[0] = map_feature[i].estimate_hx - map_feature[i].hx;
			map_feature[i].translation_error[1] = map_feature[i].estimate_hy - map_feature[i].hy;
			map_feature[i].translation_error[2] = map_feature[i].estimate_hz - map_feature[i].hz;

			map_feature[i].translation_error_magnitude = sqrt(map_feature[i].translation_error[0] * map_feature[i].translation_error[0] + map_feature[i].translation_error[1] * map_feature[i].translation_error[1] + map_feature[i].translation_error[2] * map_feature[i].translation_error[2]);			
		}
		map_feature[i].correction_outlier = false;
	}
	SearchOutlier();
	Correction();//修正平移
}

void VO::SearchOutlier()
{
	//fstream app_inlier_num("inlier_num.txt", ios::app);
	double SD_threshold = 22.0;//Standard Deviation Threshold
	double average = 0.0,SD=0.0;//標準差
	int inlier_num = 0; double SD_sum = 0.0,radius=0.0,temp=0.0;
	bool flag = 1;
	int outlier=0;
	//app_inlier_num << "PhotoCount" << PhotoCount << endl;

	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
		{
			average += map_feature[i].translation_error_magnitude;
			inlier_num++;
		}
	}
	//app_inlier_num << setw(5) << inlier_num;
	average = average / inlier_num;
	//app_inlier_num <<  "average" << average << endl;
	//計算標準差
	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
		{
			SD_sum += (average - map_feature[i].translation_error_magnitude)*(average - map_feature[i].translation_error_magnitude);
			//app_inlier_num << map_feature[i].num << setw(10) << map_feature[i].translation_error_magnitude << endl;
		}
	}
	SD = sqrt(SD_sum / inlier_num);
	//app_inlier_num << "SD" << SD << endl;
	flag = 0;
	while (SD > SD_threshold)
	{
		//app_inlier_num << PhotoCount << endl;
		inlier_num = 0; average = 0.0; SD_sum = 0.0; temp = 0.0; radius = 0.0;
		//計算平均值
		for (int i = 0; i < map_feature.size(); i++)
		{
			if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
			{
				temp = sqrt((map_feature[i].translation_error_magnitude - average)*(map_feature[i].translation_error_magnitude - average));
			}
			if (temp > radius)
			{
				radius = temp;
				outlier = i;
			}
		}
		//app_inlier_num << "------------Remove---------------" << endl;
		map_feature[outlier].correction_outlier = true;
		//app_inlier_num << map_feature[outlier].num << endl;
		flag = 1;
		if (flag)
		{
			for (int i = 0; i < map_feature.size(); i++)
			{
				if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
				{
					average += map_feature[i].translation_error_magnitude;
					inlier_num++;
				}
			}
			//app_inlier_num << "inlier_num"<<setw(5) << inlier_num<<endl;
			average = average / inlier_num;
			//app_inlier_num << "average" << average << endl;
			//計算標準差
			for (int i = 0; i < map_feature.size(); i++)
			{
				if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
				{
					SD_sum += (average - map_feature[i].translation_error_magnitude)*(average - map_feature[i].translation_error_magnitude);
					//app_inlier_num << map_feature[i].num << setw(10) << map_feature[i].translation_error_magnitude << endl;
				}
			}
			SD = sqrt(SD_sum / inlier_num);
			//app_inlier_num << "SD" << SD << endl;
		}
		
	}
	//app_inlier_num << endl;
	//app_inlier_num.close();

}

void VO::Correction()
{
	double correction_x = 0.0, correction_y = 0.0, correction_z = 0.0;
	int inlier_num = 0, cout = 0;
	Ransac_Pos temp;
	Correction_error_Pos.clear();
	feature_vec pos_temp;//rotation求重心
	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable)		cout++;
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable && !map_feature[i].correction_outlier)//在地圖且比對為成功者
		{
			inlier_num++;
			correction_x += map_feature[i].translation_error[0];
			correction_y += map_feature[i].translation_error[1];
			correction_z += map_feature[i].translation_error[2];
			pos_temp.current[0] += map_feature[i].hx;	//地標求重心
			pos_temp.current[1] += map_feature[i].hy;
			pos_temp.current[2] += map_feature[i].hz;
			pos_temp.estimate[0] += map_feature[i].estimate_hx; //此刻回推地標求重心
			pos_temp.estimate[1] += map_feature[i].estimate_hy;
			pos_temp.estimate[2] += map_feature[i].estimate_hz;
		}
	}
	temp.X[0] = correction_x / inlier_num; temp.X[1] = correction_y / inlier_num; temp.X[2] = correction_z / inlier_num;
	pos_temp.current[0] = pos_temp.current[0] / inlier_num; pos_temp.current[1] = pos_temp.current[1] / inlier_num; pos_temp.current[2] = pos_temp.current[2] / inlier_num;
	pos_temp.estimate[0] = pos_temp.estimate[0] /inlier_num; pos_temp.estimate[1] = pos_temp.estimate[1] / inlier_num; 
	pos_temp.estimate[2] = pos_temp.estimate[2] / inlier_num;

	Correction_error_Pos.push_back(temp);
	//Correction_Rotation(Correction_error_Pos, pos_temp, inlier_num);

	fstream app_correct_error("correct_error.txt", ios::app);
	fstream app_correct_pos("correct_pos.txt", ios::app);
	if (inlier_num>=cout*0.8)
	{
		VO_Pos[0].X[0] = VO_Pos[0].X[0] + inlier_num*Correction_error_Pos[0].X[0] / cout;
		VO_Pos[0].X[1] = VO_Pos[0].X[1] + inlier_num*Correction_error_Pos[0].X[1] / cout;
		VO_Pos[0].X[2] = VO_Pos[0].X[2] + inlier_num*Correction_error_Pos[0].X[2] / cout;
		//VO_Pos[0].X[3] = VO_Pos[0].X[3] + inlier_num*Correction_error_Pos[0].X[3] / cout;//+? -?
		//VO_Pos[0].X[4] = VO_Pos[0].X[4] + inlier_num*Correction_error_Pos[0].X[4] / cout;
		//VO_Pos[0].X[5] = VO_Pos[0].X[5] + inlier_num*Correction_error_Pos[0].X[5] / cout;

		app_correct_error << PhotoCount <<setw(5) << cout << setw(5) << inlier_num << setw(10) << Correction_error_Pos[0].X[0] << setw(10) << Correction_error_Pos[0].X[1] << setw(10) << Correction_error_Pos[0].X[2] << setw(10) << Correction_error_Pos[0].X[3] << setw(10) << Correction_error_Pos[0].X[4] << setw(10) << Correction_error_Pos[0].X[5] << setw(10) << endl;
		app_correct_pos << PhotoCount << setw(10) << VO_Pos[0].X[0] << setw(10) << VO_Pos[0].X[1] << setw(10) << VO_Pos[0].X[2] << endl;
	}
	app_correct_error.close();
	app_correct_pos.close();
	
}


void VO::Keyfeature_Selection() {

	match_stable_num = 0,  ray = 0, pre_new_add = 0;
	int on_image = 0;
	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image)
			on_image++;
	}
	//fstream app_frame("frame.txt", ios::app);
	for (int i = 0; i < map_feature.size(); i++)
	{
		if (map_feature[i].on_image && !map_feature[i].new_add && map_feature[i].match_stable)//在地圖且比對為成功者
		{
			//match_stable_num++;//比對成功數量增加
			double ray_vector[3];
			double dist = 0;

			ray_vector[0] = (map_feature[i].l_ix - l_u0) / l_fu * map_feature[i].depthvalue;
			ray_vector[1] = (map_feature[i].l_iy - l_v0) / l_fv * map_feature[i].depthvalue;
			ray_vector[2] = map_feature[i].depthvalue;

			dist = (ray_vector[0] - map_feature[i].r_hx)*(ray_vector[0] - map_feature[i].r_hx) + (ray_vector[1] - map_feature[i].r_hy)*(ray_vector[1] - map_feature[i].r_hy) + (ray_vector[2] - map_feature[i].r_hz)*(ray_vector[2] - map_feature[i].r_hz);
			dist = sqrt(dist);
			if (dist > 15)
				ray++;
			match_stable_num++;
		}

		if (map_feature[i].new_add)	//前一張有新增特徵點
			pre_new_add++;

	}

	if (ray > 0.3*match_stable_num || pre_new_add>on_image*0.2)//PhotoCount%2==0 && PhotoCount>=2)  //
	{
		ransac_time++;
		//Comparison_Feature_last_map_feature_descriptor( pair_feature );//比對last_map_feature 特徵
		/*app_frame << PhotoCount << endl;
		app_frame << PhotoCount << setw(8) << match_stable_num << setw(5) << ray << setw(5) <<
			pre_new_add << setw(8) << endl;
		app_frame.close();*/

		Keep_Feature temp2;

		matchfeature.clear();

		for (int j = 0; j<map_feature.size() && matchfeature.size()<20; j++)
		{
			double depth = 0.0;
			depth = abs(map_feature[j].pre_depth - map_feature[j].depthvalue);
			if (map_feature[j].on_image && !map_feature[j].new_add && depth<1000 )
			{
				memset(&temp2, 0, sizeof(Keep_Feature));
				temp2.num = map_feature[j].num;
				temp2.hx = map_feature[j].hx;
				temp2.hy = map_feature[j].hy;
				temp2.hz = map_feature[j].hz;
				temp2.l_ix = map_feature[j].l_ix;
				temp2.l_iy = map_feature[j].l_iy;
				temp2.r_ix = map_feature[j].l_ix;
				temp2.r_iy = map_feature[j].l_ix;
				temp2.depthvalue = map_feature[j].depthvalue;

				matchfeature.push_back(temp2);
			}

		}
		cout<<"matchfeature size"<<setw(5)<<matchfeature.size()<<setw(5)<<"map_feature size"<<setw(5)<<map_feature.size()<<endl;
		//fstream app_f("points.txt", ios::app);
		//app_f << matchfeature.size() << endl;
		if (matchfeature.size() >= 3)//&& PhotoCount>=2)
		{
			/*fstream app_ma("matchfeature.txt", ios::app);
			fstream app_map("mapfeature.txt", ios::app);
			app_ma << PhotoCount << endl;
			for (int i = 0; i < matchfeature.size(); i++)
			{
				app_ma << matchfeature[i].num<<setw(12)<<matchfeature[i].hx << setw(12) << matchfeature[i].hy << setw(12) << matchfeature[i].hz << setw(12) << matchfeature[i].l_ix << setw(12) << matchfeature[i].l_iy << setw(12) << matchfeature[i].depthvalue << endl;
			}
			app_ma << endl;
			app_ma.close();

			app_map << PhotoCount << endl;
			for (int i = 0; i < map_feature.size(); i++)
			{
					app_map << map_feature[i].num << setw(12) << map_feature[i].hx << setw(12) << map_feature[i].hy << setw(12) << map_feature[i].hz << setw(12) << map_feature[i].l_ix << setw(12) << map_feature[i].l_iy << setw(12) << map_feature[i].depthvalue << endl;
			}
			app_map << endl;
			app_map.close();*/

			Gpu_mapfeature.clear();
			
			for (int j = 0; j<map_feature.size(); j++)
			{
				if (map_feature[j].on_image)
				{
					memset(&temp2, 0, sizeof(Keep_Feature));
					temp2.num = map_feature[j].num;
					temp2.hx = map_feature[j].hx;
					temp2.hy = map_feature[j].hy;
					temp2.hz = map_feature[j].hz;
					temp2.l_ix = map_feature[j].l_ix;
					temp2.l_iy = map_feature[j].l_iy;
					temp2.r_ix = map_feature[j].l_ix;
					temp2.r_iy = map_feature[j].l_ix;
					temp2.depthvalue = map_feature[j].depthvalue;

					Gpu_mapfeature.push_back(temp2);
				}

			}
			
			link_RANSAC.Run_RANSAC(matchfeature, map_feature, PhotoCount, l_u0,l_v0,l_fu, l_fv);  //比對到特徵大於三個才做RANSAC
			
			/* double *CameraPose_dub;
			float *CameraPose;
			CameraPose_dub=(double*)malloc(12*sizeof(double));
			CameraPose=(float*)malloc(12*sizeof(float));
			
			int max_consensus;
			int map_num_threshold = map_feature.size()*0.5;
			//int threshold =6;//map_feature.size()*0.1;
			if (matchfeature.size() < 10)	map_num_threshold = 4;
			max_consensus=launchVOKernel(matchfeature,Gpu_mapfeature,CameraPose,&max_consensus,l_u0,l_v0,l_fu,l_fv);
			//launchVOKernel_dub(matchfeature,map_feature,CameraPose_dub,l_u0,l_v0,l_fu,l_fv);
			cout<<"max_consensus"<<max_consensus<<endl;
			if(max_consensus>map_num_threshold)
			{
				Ransac_Pos GPU_Pos;
				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						GPU_Pos.Rs[i][j]=CameraPose[i*3+j+3];
					}
					GPU_Pos.X[i]=CameraPose[i];
				}
			
				cout<<"--------------------GPU Camera Pose-----------------------"<<endl;
				for(int i=0;i<3;i++)
					printf("%.13f\n",GPU_Pos.X[i]);

				printf("%.13f\n",atan2(GPU_Pos.Rs[1][2], GPU_Pos.Rs[2][2])* 180 / PI + 90);
				printf("%.13f\n",asin(-GPU_Pos.Rs[0][2])* 180 / PI);
				printf("%.13f\n",atan2(GPU_Pos.Rs[0][1], GPU_Pos.Rs[0][0])* 180 / PI);
				cout<<"--------------------GPU Camera Pose-----------------------"<<endl;
			} */
			
			/* cout<<ransac_match[0].X[3]* 180 / PI + 90<<endl;
			cout << ransac_match[0].X[4] * 180 / PI <<endl;
			cout << ransac_match[0].X[5] * 180 / PI << endl; */
			
			//cout<<GPU_Pos.X[0]<<setw(10)<<GPU_Pos.X[1]<<setw(10)<<GPU_Pos.X[2]<<endl;
			//if (link_RANSAC.average_ransac_match.size())	/*	VO_Pos = link_RANSAC.ransac_match;*/
			//{
			//	VO_Pos = link_RANSAC.average_ransac_match;
			//	//if (map_feature.size() && PhotoCount>10)		Calculate_Landmark_Current();
			//}
			//else if (link_RANSAC.ransac_match.size())
			//{
			//	VO_Pos = link_RANSAC.ransac_match;
			//	//if (map_feature.size() && PhotoCount>10)		Calculate_Landmark_Current();
			//}
			if (link_RANSAC.ransac_match.size())	/*	VO_Pos = link_RANSAC.ransac_match;*/
			{
				VO_Pos = link_RANSAC.ransac_match;
				if (map_feature.size() && PhotoCount>60)		Calculate_Landmark_Current();
			}

			for (int i = 0; i < map_feature.size(); i++)
			{
				if (map_feature[i].on_image && !map_feature[i].new_add)
				{
					map_feature[i].r_hx = (map_feature[i].l_ix - l_u0) / l_fu * map_feature[i].depthvalue;
					map_feature[i].r_hy = (map_feature[i].l_iy - l_v0) / l_fv * map_feature[i].depthvalue;
					map_feature[i].r_hz = map_feature[i].depthvalue;
				}

			}

		}

	}
	
}


void VO::MatchUpdate_FromGPU_Result(vector<PairFeature> &pair_feature,vector<int>&gpu_matcher)
{
	cout<<"gpu_matcher size"<<setw(5)<<gpu_matcher.size()/2<<endl;
	
	for(int i=0;i<gpu_matcher.size()/2;i++)
	{
		/* int map_index=gpu_matcher[2*i];
		int img_index=gpu_matcher[2*i+1]; */
		int map_index=gpu_matcher[2*i+1];
		int img_index=gpu_matcher[2*i];
	
		if (map_feature[map_index].times<=10) //十次
				{
					map_feature[map_index].times++;

                    if ((fabs(map_feature[map_index].l_ix-pair_feature[img_index].ix)<15) && (fabs(map_feature[map_index].l_iy-pair_feature[img_index].iy)<15))
					{
							map_feature[map_index].filter<<=1;
							map_feature[map_index].filter++;
							
					}
					else
						map_feature[map_index].filter<<=1;
				}
				if (map_feature[map_index].times>=10)
				{
					map_feature[map_index].filter=map_feature[map_index].filter %256;  //限定大小

						if ((fabs(map_feature[map_index].l_ix-pair_feature[img_index].ix)<15) && (fabs(map_feature[map_index].l_iy-pair_feature[img_index].iy)<15))
						{
							map_feature[map_index].filter<<=1;
							map_feature[map_index].filter++;
						}
						else
						{
						map_feature[map_index].filter<<=1;
						}
                        map_feature[map_index].stable=0;
						for(int j=0;j<8;j++) //計算八次內穩定幾次
						{
							if ((map_feature[map_index].filter>>j)&1==1)
								map_feature[map_index].stable++;
						}
						if (map_feature[map_index].stable<5)
							map_feature[map_index].count++;  //不穩定計數+1
				}

				//if ((map_feature[i].depthvalue - map_feature[i].pre_depth) < map_feature[i].depthvalue / 10 && PhotoCount>10)	continue;
					map_feature[map_index].l_ix_pre = map_feature[map_index].l_ix;
					map_feature[map_index].l_iy_pre = map_feature[map_index].l_iy;
                	map_feature[map_index].l_ix = pair_feature[img_index].ix;
				    map_feature[map_index].l_iy = pair_feature[img_index].iy;
			//	    map_feature[i].r_ix = pair_feature[match_num].r_ix;
			//	    map_feature[i].r_iy = pair_feature[match_num].r_iy;
					map_feature[map_index].dir = pair_feature[img_index].dir;
					map_feature[map_index].size = pair_feature[img_index].size;
					map_feature[map_index].hessian = pair_feature[img_index].hessian;
					map_feature[map_index].laplacian = pair_feature[img_index].laplacian;
					map_feature[map_index].appear = PhotoCount;
					map_feature[map_index].pre_depth = map_feature[map_index].depthvalue;
					map_feature[map_index].depthvalue=pair_feature[img_index].depthvalue;
					map_feature[map_index].on_image = true;

				map_feature[map_index].match = true;
				pair_feature[img_index].match = true;
				//描述向量更新
				for(int j=0 ; j<dDscpt ; j++)
				{
					map_feature[map_index].original_descriptor[j] = (map_feature[map_index].original_descriptor[j]+pair_feature[img_index].descriptor[j])/2;
				}
				//
				for(int j=0 ; j<dDscpt ; j++)
				{
					map_feature[map_index].current_descriptor[j] = pair_feature[img_index].descriptor[j];
				}
		
		
	}
	
	
}

void VO::Matching_CPU(vector<Keep_Feature> &map_feature, vector<PairFeature> &pair_feature,vector<int>&cpu_matcher)
{
	
	for(int i=0 ; i<map_feature.size() ; i++)
		{
			if(map_feature[i].match) continue;

			double dist=1e6;
			int match_num=0 , match_num2=0; 

			for(int j=0 ; j<pair_feature.size() ; j++)
			{				
				if( map_feature[i].laplacian == pair_feature[j].laplacian )
				{
					double dist1=0;
					for(int k=0 ; k<dDscpt ; k++)
					{
						dist1 += (map_feature[i].original_descriptor[k] - pair_feature[j].descriptor[k])*(map_feature[i].original_descriptor[k] - pair_feature[j].descriptor[k]);
					}

					dist1=sqrt(dist1);

					if(dist1 < dist)
					{
						dist = dist1;
						match_num = j;
					}
				}
			}
			
			if(dist > 1e5) continue;

			dist=1e6;
			for(int j=0 ; j<map_feature.size() ; j++)
			{
				if( map_feature[j].laplacian == pair_feature[match_num].laplacian )
				{
					double dist1=0;
					for(int k=0 ; k<dDscpt ; k++)
					{
						dist1 += (map_feature[j].original_descriptor[k] - pair_feature[match_num].descriptor[k])*(map_feature[j].original_descriptor[k] - pair_feature[match_num].descriptor[k]);
					}

					dist1=sqrt(dist1);

					if(dist1 < dist)
					{
						dist = dist1;
						match_num2 = j;
					}
				}
			}

			if( i==match_num2 && dist<0.09 )
			{
				cpu_matcher.push_back(i);
				cpu_matcher.push_back(match_num);
				if (map_feature[i].times<=10) //十次
				{
					map_feature[i].times++;

                    if ((fabs(map_feature[i].l_ix-pair_feature[match_num].ix)<15) && (fabs(map_feature[i].l_iy-pair_feature[match_num].iy)<15))
					{
							map_feature[i].filter<<=1;
							map_feature[i].filter++;
							
					}
					else
						map_feature[i].filter<<=1;
				}
				if (map_feature[i].times>=10)
				{
					map_feature[i].filter=map_feature[i].filter %256;  //限定大小

						if ((fabs(map_feature[i].l_ix-pair_feature[match_num].ix)<15) && (fabs(map_feature[i].l_iy-pair_feature[match_num].iy)<15))
						{
							map_feature[i].filter<<=1;
							map_feature[i].filter++;
						}
						else
						{
						map_feature[i].filter<<=1;
						}
                        map_feature[i].stable=0;
						for(int j=0;j<8;j++) //計算八次內穩定幾次
						{
							if ((map_feature[i].filter>>j)&1==1)
								map_feature[i].stable++;
						}
						if (map_feature[i].stable<5)
							map_feature[i].count++;  //不穩定計數+1
				}

				//if ((map_feature[i].depthvalue - map_feature[i].pre_depth) < map_feature[i].depthvalue / 10 && PhotoCount>10)	continue;
					map_feature[i].l_ix_pre = map_feature[i].l_ix;
					map_feature[i].l_iy_pre = map_feature[i].l_iy;
                	map_feature[i].l_ix = pair_feature[match_num].ix;
				    map_feature[i].l_iy = pair_feature[match_num].iy;
			//	    map_feature[i].r_ix = pair_feature[match_num].r_ix;
			//	    map_feature[i].r_iy = pair_feature[match_num].r_iy;
					map_feature[i].dir = pair_feature[match_num].dir;
					map_feature[i].size = pair_feature[match_num].size;
					map_feature[i].hessian = pair_feature[match_num].hessian;
					map_feature[i].laplacian = pair_feature[match_num].laplacian;
					map_feature[i].appear = PhotoCount;
					map_feature[i].pre_depth = map_feature[i].depthvalue;
					map_feature[i].depthvalue=pair_feature[match_num].depthvalue;
					map_feature[i].on_image = true;

				map_feature[i].match = true;
				pair_feature[match_num].match = true;
				//描述向量更新
				for(int j=0 ; j<dDscpt ; j++)
				{
					map_feature[i].original_descriptor[j] = (map_feature[i].original_descriptor[j]+pair_feature[match_num].descriptor[j])/2;
				}
				//
				for(int j=0 ; j<dDscpt ; j++)
				{
					map_feature[i].current_descriptor[j] = pair_feature[match_num].descriptor[j];
				}
			}
		}
}

void VO::Save_Feature( vector<PairFeature> &pair_feature , vector<int>& add_map_feature )
{
	on_image_num=0;
	
	

    for (int i=0;i<3;i++)
        location[i]=0;

	//搜尋影像上特徵分區
	for(int i=0 ; i<pair_feature.size() ; i++)
	{
		if(pair_feature[i].match)
		{
			on_image_num++;
		
		if(int(pair_feature[i].ix/213)==0)
			location[0]++;
		else if(int(pair_feature[i].ix/213)==1)
			location[1]++;
		else location[2]++;
		}
	}
   	for (int g=0;g<3;g++)
	{
			if( location[g]<7 )
			{
				Keep_Feature temp;

				for(int i=0 ; i<pair_feature.size() && location[g]<7 ; i++) //|| !(location1>0 &&location2 >0 && location3>0 )
				{
					if (int(pair_feature[i].ix / 213) == g || PhotoCount == 0)
					{
						bool is_near = false, is_similar = false, locat = false;
						for (int j = 0; j < map_feature.size(); j++)
						{
							////判斷特徵點是否太相近

							if (((fabs(pair_feature[i].ix - map_feature[j].l_ix) <= (add_window_size - 1) / 2 &&
								fabs(pair_feature[i].iy - map_feature[j].l_iy) <= (add_window_size - 1) / 2))
								)
							{
								is_near = true;
							}

							double dist = 1e6;

							for (int k = 0; k < dDscpt; k++)
							{
								dist += (pair_feature[i].descriptor[k] - map_feature[j].original_descriptor[k])*(pair_feature[i].descriptor[k] - map_feature[j].original_descriptor[k]);
							}

							dist = sqrt(dist);

							if (dist < similar_threshold) is_similar = true;

						}

						//ix位置
						if (int(pair_feature[i].ix / 213) == g) locat = true;
						double hz = 0, hy = 0, hx = 0;
						if (!is_similar && !is_near && !pair_feature[i].match  && locat)	//不相似也不靠近
						{

							memset(&temp, 0, sizeof(Keep_Feature));
							temp.l_ix = pair_feature[i].ix;
							temp.l_iy = pair_feature[i].iy;
							//	temp.r_ix = pair_feature[i].r_ix;
							//	temp.r_iy = pair_feature[i].r_iy;
							temp.hessian = pair_feature[i].hessian;
							temp.laplacian = pair_feature[i].laplacian;
							temp.size = pair_feature[i].size;
							temp.dir = pair_feature[i].dir;
							temp.new_add = true;
							temp.on_image = true;
							temp.depthvalue = pair_feature[i].depthvalue;
							pair_feature[i].pre_depth = 0;

							for (int j = 0; j < dDscpt; j++)
							{
								temp.original_descriptor[j] = pair_feature[i].descriptor[j];
							}

							temp.num = feature_num;
							feature_num++;
							on_image_num++;
							location[g]++;

							hz = pair_feature[i].depthvalue;
							hx = hz*(pair_feature[i].ix - l_u0) / l_fu;
							hy = hz*(pair_feature[i].iy - l_v0) / l_fv;


							temp.hx = VO_Pos[0].Rs[0][0] * hx + VO_Pos[0].Rs[0][1] * hy + VO_Pos[0].Rs[0][2] * hz + VO_Pos[0].X[0];

							temp.hy = VO_Pos[0].Rs[1][0] * hx + VO_Pos[0].Rs[1][1] * hy + VO_Pos[0].Rs[1][2] * hz + VO_Pos[0].X[1];

							temp.hz = VO_Pos[0].Rs[2][0] * hx + VO_Pos[0].Rs[2][1] * hy + VO_Pos[0].Rs[2][2] * hz + VO_Pos[0].X[2];


							//				double feature_x, feature_y, feature_z;
											///////////////////SSG



							add_map_feature.push_back((int)map_feature.size()); // 要新增的地圖特徵點編號
							map_feature.push_back(temp);



						}

					}

				}
			}

	}
	
}


void VO::Comparison_Feature_original_descriptor( vector<PairFeature> &pair_feature)// ,bool isMoveDone,bool isHeadingDone)
{
	if( map_feature.size() && pair_feature.size() )
	{
		//********************Matching with Cpu********************//
		vector<int>cpu_matcher;
		cpu_matcher.clear();
		Matching_CPU(map_feature,pair_feature,cpu_matcher);
		for(int i=0;i<cpu_matcher.size()/2;i++)
			cout<<"map_feature    "<<cpu_matcher[2*i]<<"pair_feature    "<<cpu_matcher[2*i+1]<<endl;
		cout<<"pair_feature size  "<<pair_feature.size()<<endl;
		
		
		//********************Matching with Gpu********************//block開的數量為pair_feature
		/* vector<int>gpu_matcher;
		gpu_matcher.clear();
		//launchMatchKernel(pair_feature.size(),map_feature.size(),dDscpt,pair_feature,map_feature,gpu_matcher);
		//launchMatchKernel(map_feature.size(),pair_feature.size(),dDscpt,map_feature,pair_feature,gpu_matcher);
		launchMatchKernel(pair_feature.size(),map_feature.size(),dDscpt,map_feature,pair_feature,gpu_matcher);
		MatchUpdate_FromGPU_Result(pair_feature,gpu_matcher); */
		/* for(int i=0;i<gpu_matcher.size()/2;i++)
			cout<<"map_feature    "<<gpu_matcher[2*i+1]<<"pair_feature    "<<gpu_matcher[2*i]<<endl; */
	}
}


void VO::Comparison_Feature( vector<SingleFeature>&l_feature , vector<PairFeature>&pair_feature )//, bool isMoveDone,bool isHeadingDone)
{
	//fstream app_last_map_feature("last_map_feature.txt",ios::app);
	//match_stable_num = 0, ray = 0, pre_new_add = 0;
	vector<int>key_frame;
	key_frame.clear();
	for(int i=0 ; i<map_feature.size() ; i++)
	{
		map_feature[i].match = false;
	}

	if( l_feature.size() )
	{
		
		datamatch.Search_Pair_Feature( l_feature, pair_feature);

		Comparison_Feature_original_descriptor( pair_feature);//, isMoveDone,isHeadingDone );

		Keyfeature_Selection();
		
	}
	/* fstream app_pose("pose.txt", ios::app);
	app_pose << setw(3) << PhotoCount << setw(5) << map_feature.size();
	//app_x<<setw(15)<<ransac_match[u].match_set.size();
	app_pose << setw(12) << VO_Pos[0].X[0] << setw(12) << VO_Pos[0].X[1] << setw(12) << VO_Pos[0].X[2] << setw(12);
	app_pose << VO_Pos[0].X[3] * 180 / PI + 90 << setw(12);
	app_pose << VO_Pos[0].X[4] * 180 / PI << setw(12);
	app_pose << VO_Pos[0].X[5] * 180 / PI << endl;
	app_pose.close(); */
}



void VO::Find_Feature(const Mat l_image , vector<SingleFeature>& l_feature , vector<int>& find_map_feature)
{
	for( int i=0 ; i < (int)map_feature.size() ; i++ )
	{
		//map_feature[i].old_match = false;
		map_feature[i].on_image = false;
	}
	
	SURFParams params;
	params.hessianThreshold=hessian_threshold;
	params.nOctaves=octaves;
	params.nOctavelayers=octave_layers;
	
	map.GetFeatureData(l_image,params,l_feature);
	sort( l_feature.begin(), l_feature.end(), Hessian_StrengthVO ); // hessian值由大到小排序
	
	for( int i=0 ; i < (int)l_feature.size() ; i++ )
	{
		map.Image_Correction( l_feature[i].ix, l_feature[i].iy, l_fu, l_fv, l_u0, l_v0, l_coefficient );
	}
	
	found = false;

	lost = false;
	if( (int)map_feature.size()  &&  !on_image_num )   lost = true;
}


void VO::Run_VO( const Mat l_image, const double sample_time,int PhotoCount)
{
	vector<int> erase_map_feature;//建立整數空間的vector,使用名為erase_map_feature,erase_map_feature[i]中皆為整數
	erase_map_feature.clear();//將vector<int>向量進行清空初始化

	vector<int> erase_disappear_map_feature;//建立整數空間的vector,使用名為erase_disappear_map_feature,erase_disappear_map_feature[i]中皆為整數
	erase_disappear_map_feature.clear();//將vector<int>向量進行清空初始化

	vector<int> add_map_feature;//建立整數空間的vector,使用名為add_map_feature,add_map_feature[i]中皆為整數
	add_map_feature.clear();//將vector<int>向量進行清空初始化

	vector<int> find_map_feature;//建立整數空間的vector,使用名為find_map_feature,find_map_feature[i]中皆為整數
	find_map_feature.clear();//將vector<int>向量進行清空初始化

	fstream app_cp("correct_position.txt", ios::app);
	
	vector<SingleFeature> l_feature;
	l_feature.clear();
	vector<PairFeature> pair_feature;
	pair_feature.clear();
	
	
	
	map.Erase_Bad_Feature(map_feature,erase_map_feature , erase_disappear_map_feature,on_image_num,max_on_image_num,erase_times,erase_ratio);
	
	Find_Feature(l_image,l_feature,find_map_feature);
	
	Comparison_Feature( l_feature , pair_feature );
	
	for (int i = 0; i<(int)map_feature.size(); i++)
			map_feature[i].new_add = false;
		
	if(link_RANSAC.ransac_match.size() || PhotoCount==0 || link_RANSAC.average_ransac_match.size())
			Save_Feature( pair_feature , add_map_feature );	
	
	VO_Set_Hessian_Threshold(hessian_threshold, on_image_num);
	
	app_cp << setw(3) << PhotoCount << setw(5) << map_feature.size();
	//app_x<<setw(15)<<ransac_match[u].match_set.size();
	app_cp << setw(12) << VO_Pos[0].X[0] << setw(12) << VO_Pos[0].X[1] << setw(12) << VO_Pos[0].X[2] << setw(12);
	app_cp << VO_Pos[0].X[3] * 180 / PI + 90 << setw(12);
	app_cp << VO_Pos[0].X[4] * 180 / PI << setw(12);
	app_cp << VO_Pos[0].X[5] * 180 / PI << endl;
	app_cp.close();
	
	cout<<"PhotoCount"<<PhotoCount<<setw(5)<<"map_feature size"<<map_feature.size()<<endl;
	///////////test////////
/* 	for(int i=0;i<map_feature.size();i++)
		cout<< map_feature[i].num << setw(12) << map_feature[i].hx << "  " << setw(8) << map_feature[i].hy << "  " << setw(10) << map_feature[i].hz << setw(10) << map_feature[i].l_ix<< setw(10) << map_feature[i].l_iy<< setw(10) << map_feature[i].pre_depth<< endl;
	 */
	
	
	l_feature.clear();
	pair_feature.clear();
	add_map_feature.clear();
	erase_map_feature.clear();
	erase_disappear_map_feature.clear();
	find_map_feature.clear();
	
	l_feature.swap(l_feature);
	pair_feature.swap(pair_feature);
	add_map_feature.swap(add_map_feature);
	erase_map_feature.swap(erase_map_feature);
	erase_disappear_map_feature.swap(erase_disappear_map_feature);
	find_map_feature.swap(find_map_feature);
	
	
	
	//PhotoCount++;
	
}