#include "RANSAC.h"


void RANSAC_ho::Run_RANSAC( vector<Keep_Feature> matchfeature , vector<Keep_Feature> map_feature,int PhotoCount,double l_u0,double l_v0,double l_fu,double l_fv)
{
	bool again = true; 
	double range = 25.0;
	int match_number ;
	int maxsetsize=0;
	int match_max = 0;
	for(int i=0;i<map_feature.size();i++)
		if(map_feature[i].on_image)
			cout<< map_feature[i].num << setw(12) << map_feature[i].hx << "  " << setw(8) << map_feature[i].hy << "  " << setw(10) << map_feature[i].hz << setw(10) << map_feature[i].l_ix<< setw(10) << map_feature[i].l_iy<< setw(10) << map_feature[i].pre_depth<< endl;
	cout<<endl<<endl;
	
	
	for(int i=0;i<matchfeature.size();i++)
		cout<< matchfeature[i].num << setw(12) << matchfeature[i].hx << "  " << setw(8) << matchfeature[i].hy << "  " << setw(10) << matchfeature[i].hz << setw(10) << matchfeature[i].l_ix<< setw(10) << matchfeature[i].l_iy<< setw(10) << matchfeature[i].depthvalue<< endl;
	
	//l_u0=para[0];l_v0=para[1];l_fu=para[2];l_fv=para[3];
	int map_num_threshold = map_feature.size()*0.5;
	//int threshold =6;//map_feature.size()*0.1;
	if (matchfeature.size() < 10)	map_num_threshold = 4;
	int num=0,times=0;
	//vector<double> ransac_error;
	//ransac_error.clear();
	cout<<"map_num_threshold"<<setw(10)<<map_num_threshold<<endl;
	struct timeval starttime, endtime;
	double executime;
	gettimeofday(&starttime, NULL);
	for(int g=0; again && g<matchfeature.size()-2 ; g++)
	{
		for(int g_two=g+1 ; again && g_two< matchfeature.size()-1 ; g_two++ )
		{
			for(int g_three=g_two+1 ; again && g_three<matchfeature.size() ; g_three++ )
			{
				vector<Matrix> X; 
				X.clear();
				vector< Rectangular > Rs;
				Rs.clear();
				vector< Matrix > ts;
				ts.clear();
				//times++;
				vector<int> true_match_number;
				true_match_number.clear();

				char k = 0;
				double ptw[9];
				double pti[6];
				double tmp[5];
				//IxIy temp;

				ptw[0] = matchfeature[g].hx;//-741.069;
				ptw[1] = matchfeature[g].hy;//2212.00;
				ptw[2] = matchfeature[g].hz;//-217.026;
				pti[0] = matchfeature[g].l_ix;//149.686;
 				pti[1] = matchfeature[g].l_iy;//7778521.000;

				ptw[3] = matchfeature[g_two].hx;//-603.6699;
				ptw[4] = matchfeature[g_two].hy;//2812.00;
				ptw[5] = matchfeature[g_two].hz;//203.2188;
				pti[2] = matchfeature[g_two].l_ix;//206.002;
 				pti[3] = matchfeature[g_two].l_iy;//0.000;

				ptw[6] = matchfeature[g_three].hx;//-921.0765;
				ptw[7] = matchfeature[g_three].hy;//2883.00;
				ptw[8] = matchfeature[g_three].hz;//440.7917;
				pti[4] = matchfeature[g_three].l_ix;//2859.73800;
 				pti[5] = matchfeature[g_three].l_iy;//1.090e267;
				
				/* ptw[0] = -905.204;//-741.069;
				ptw[1] = 2835;//2212.00;
				ptw[2] = 353.502;//-217.026;
				pti[0] = 149.807;//149.686;
 				pti[1] = 178.231;//7778521.000;

				ptw[3] = -1088.31;//-603.6699;
				ptw[4] = 2212;//2812.00;
				ptw[5] = -561.739;//203.2188;
				pti[2] = 58.3012;//206.002;
 				pti[3] = 378.804;//0.000;

				ptw[6] = -741.062;//-921.0765;
				ptw[7] = 2212;//2883.00;
				ptw[8] = -217.046;//440.7917;
				pti[4] = 141.455;//2859.73800;
 				pti[5] = 296.255;//1.090e267; */
				
				double dep[3];
				dep[0] = matchfeature[g].depthvalue * matchfeature[g].depthvalue;
				dep[1] = matchfeature[g_two].depthvalue * matchfeature[g_two].depthvalue;
				dep[2] = matchfeature[g_three].depthvalue * matchfeature[g_three].depthvalue;
				
				/* double dep[3];
				dep[0] = 2835*2835;
				dep[1] = 2212*2212;
				dep[2] = 2212*2212; */
				
				double aa = sqrt( ((pti[0] - l_u0)/l_fu)*((pti[0] - l_u0)/l_fu)*dep[0] + ((pti[1]-l_v0)/l_fv)*((pti[1]-l_v0)/l_fv)*dep[0] + dep[0]);
				double bb = sqrt( ((pti[2] - l_u0)/l_fu)*((pti[2] - l_u0)/l_fu)*dep[1] + ((pti[3]-l_v0)/l_fv)*((pti[3]-l_v0)/l_fv)*dep[1] + dep[1]);
				double cc = sqrt( ((pti[4] - l_u0)/l_fu)*((pti[4] - l_u0)/l_fu)*dep[2] + ((pti[5]-l_v0)/l_fv)*((pti[5]-l_v0)/l_fv)*dep[2] + dep[2]);
				
				/* aa=	2328.2796;
				bb= 2859.7380;
				cc= 3007.7040;
				l_u0 = 318.9861;
				l_v0 = 244.2651;
				l_fu = 529.7137;
				l_fv = 529.7751; */
				P3PSolver p3p;

				p3p.SetPointsCorrespondance_shen(ptw, pti);
				
				
				
				p3p.Solve_one_shen(X, Rs, aa, bb, cc,l_u0,l_v0,l_fu,l_fv);
				
				
				//cout<<"P3PSolver pass"<<endl;
				
				match_number = 0;
				
				/* CPU_WL[times*3]=X[0].data_[0];
				CPU_WL[times*3+1]=X[0].data_[1];
				CPU_WL[times*3+2]=X[0].data_[2]; */
				
				/* CPU_Rs[times*9]=Rs[0].data_[0][0];		CPU_Rs[times*9+3]=Rs[0].data_[1][0];	CPU_Rs[times*9+6]=Rs[0].data_[2][0];
				CPU_Rs[times*9+1]=Rs[0].data_[0][1];	CPU_Rs[times*9+4]=Rs[0].data_[1][1];	CPU_Rs[times*9+7]=Rs[0].data_[2][1];
				CPU_Rs[times*9+2]=Rs[0].data_[0][2];	CPU_Rs[times*9+5]=Rs[0].data_[1][2];	CPU_Rs[times*9+8]=Rs[0].data_[2][2];
				 */
				
				for(int i=0 ; i<X.size() ; i++)
				{
					
					vector<int> true_number;
					true_number.clear();
					true_number.push_back(matchfeature[g].num);
					true_number.push_back(matchfeature[g_two].num);
					true_number.push_back(matchfeature[g_three].num);
					int match = 0;
				
					double rx = X[i].data_[0];
					double ry = X[i].data_[1];
					double rz = X[i].data_[2];


					int setsize=0;
					
					for(int j=0 ; j<map_feature.size() ; j++)
					{
						if (map_feature[j].on_image)
						{
							 /* if (matchfeature[g].num == map_feature[j].num ||
								matchfeature[g_two].num == map_feature[j].num ||
								matchfeature[g_three].num == map_feature[j].num) continue;  */

							double feature[3], h[3];//, h2[3];
							feature[0] = (map_feature[j].l_ix - l_u0) / l_fu * map_feature[j].depthvalue;
							feature[1] = (map_feature[j].l_iy - l_v0) / l_fv * map_feature[j].depthvalue;
							feature[2] = map_feature[j].depthvalue;

							h[0] = Rs[i].data_[0][0] * feature[0] + Rs[i].data_[1][0] * feature[1] + Rs[i].data_[2][0] * feature[2];
							h[1] = Rs[i].data_[0][1] * feature[0] + Rs[i].data_[1][1] * feature[1] + Rs[i].data_[2][1] * feature[2];
							h[2] = Rs[i].data_[0][2] * feature[0] + Rs[i].data_[1][2] * feature[1] + Rs[i].data_[2][2] * feature[2];

							double dist = 0;
							
							//dist = (temp.data[0] - matchfeature[j].l_ix)*(temp.data[0] - matchfeature[j].l_ix)+(temp.data[1] - matchfeature[j].l_iy)*(temp.data[1] - matchfeature[j].l_iy);
							dist = (h[0] - map_feature[j].hx + rx)*(h[0] - map_feature[j].hx + rx) +
								(h[1] - map_feature[j].hy + ry)*(h[1] - map_feature[j].hy + ry) +
								(h[2] - map_feature[j].hz + rz)*(h[2] - map_feature[j].hz + rz);
							dist = sqrt(dist);



							if (dist < range)
							{
								true_number.push_back(map_feature[j].num);
								setsize++;//符合條件的特徵點個數
							}
						}
					}
					
					/* CPU_ransac_match[times]=setsize; */
					
					//ransac_error.push_back(error_sum);
					if(setsize==0) continue;
					if (setsize>maxsetsize) maxsetsize=setsize; //計算最大集合個數
					
					
					if(match_max<true_number.size())
					{
						true_match_number.clear();
						true_match_number = true_number;
						match_max = true_match_number.size();
						match_number = i;
						
					}
					true_number.clear();
					true_number.swap( true_number);
				}
				times++;
				
				
				if( match_max > map_num_threshold)
				{
					Ransac_Pos ransac_temp;

					ransac_temp.match_set = true_match_number;
					for (int p=0;p<3;p++)
					{
						for(int o=0;o<3;o++)
						{
						  ransac_temp.Rs[p][o]=Rs[match_number].data_[o][p];
						}
						ransac_temp.X[p]=X[match_number].data_[p];
					}
					ransac_match.push_back(ransac_temp);


				}
				X.clear();
				Rs.clear();
				true_match_number.clear();
				X.swap(X);
				Rs.swap(Rs);
				//ts.swap(vector< Matrix >());
				true_match_number.swap(true_match_number);
			}


		}


	}
	
	gettimeofday(&endtime, NULL);
				
	executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
	executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
	printf("CPU time: %13lf msec\n", executime);
	
	//fstream app_c("consensus.txt", ios::app);
	//fstream app_cc("zz.txt", ios::app);
	vector<Ransac_Pos> ransac_temp1;
    ransac_temp1=ransac_match;
	ransac_match.clear();
	for(int q=0;q<ransac_temp1.size();q++) //只存最大集合
	{
		if (ransac_temp1[q].match_set.size() == maxsetsize + 3)
		{
			ransac_match.push_back(ransac_temp1[q]);
		}
	}
	/* for (int j = 0; j < ransac_match.size(); j++)
	{
		app_c << PhotoCount;
		for (int i = 0; i < ransac_match[j].match_set.size(); i++)
		{
			app_c << setw(5) << ransac_match[j].match_set[i];
		}
		app_c << endl;
	}
	app_c.close();

	//********計算inlier次數**********目前還沒用到//
	int consensus = 0;

	if (ransac_match.size())
	{
		for (int k = 0; k < ransac_match[0].match_set.size(); k++)
		{
			for (int i = 0; i < map_feature.size(); i++)
			{
				if (ransac_match[0].match_set[k] == map_feature[i].num)
				{
					consensus = i;
					break;
				}
			}
			map_feature[consensus].consensus_time++;
		}
	} */
		
	//********計算inlier次數**********目前還沒用到//
	ransac_temp1.clear();
	ransac_temp1.swap(ransac_temp1);
	fstream app_x("position.txt",ios::app);

	//app_x << setw(3) << PhotoCount;//<<setw(12)<<"match size" << setw(12)<<"x" << setw(12) << "y" << setw(12) << "z" << setw(10) << "degree"<<endl;
	Ransac_Pos average;
	double average_X[3] = { 0 }, average_Rs[3][3] = { 0 }, average_degree[3] = {0};
	bool average_flag=false;

	if (ransac_match.size())
	{
		if (ransac_match.size() > 1)	average_flag = true;
		for (int u = 0; u < ransac_match.size(); u++)
		{
			ransac_match[u].X[3] = atan2(ransac_match[u].Rs[2][1], ransac_match[u].Rs[2][2]);
			ransac_match[u].X[4] = asin(-ransac_match[u].Rs[2][0]);
			ransac_match[u].X[5] = atan2(ransac_match[u].Rs[1][0], ransac_match[u].Rs[0][0]);

			if (!average_flag)
			{
				app_x << setw(3) << PhotoCount << setw(5) << map_feature.size();
				//app_x<<setw(15)<<ransac_match[u].match_set.size();
				app_x << setw(12) << ransac_match[u].X[0] << setw(12) << ransac_match[u].X[1] << setw(12) << ransac_match[u].X[2] << setw(12);
				app_x << ransac_match[u].X[3] * 180 / PI + 90 << setw(12);
				app_x << ransac_match[u].X[4] * 180 / PI << setw(12);
				app_x << ransac_match[u].X[5] * 180 / PI << endl;
			}
			

			if (average_flag)
			{
				for (int i = 0; i < 3; i++)
				{
					average_X[i] += ransac_match[u].X[i];
					average_degree[i] += ransac_match[u].X[3+i];
					for (int j = 0; j < 3; j++)
					{
						average_Rs[i][j] += ransac_match[u].Rs[i][j];
					}
				}
			}
		}
		cout<<"--------------------CPU Camera Pose-----------------------"<<endl;
		for(int i=0;i<3;i++)
			printf("%.13f\n",ransac_match[0].X[i]);
			//cout<<ransac_match[0].X[i]<<endl;
			
			printf("%.13f\n",ransac_match[0].X[3]* 180 / PI + 90);
			printf("%.13f\n",ransac_match[0].X[4] * 180 / PI);
			printf("%.13f\n",ransac_match[0].X[5] * 180 / PI);
			/* cout<<ransac_match[0].X[3]* 180 / PI + 90<<endl;
			cout << ransac_match[0].X[4] * 180 / PI <<endl;
			cout << ransac_match[0].X[5] * 180 / PI << endl; */
		cout<<"--------------------CPU Camera Pose-----------------------"<<endl;
	}
	
	/* if (average_flag)
	{
		for (int i = 0; i < 3; i++)
		{
			average.X[i] = average_X[i] / ransac_match.size();
			average.X[3+i]= average_degree[i] / ransac_match.size();
			for (int j = 0; j < 3; j++)
			{
				average.Rs[i][j]= average_Rs[i][j] / ransac_match.size();
			}
		}
		average_ransac_match.push_back(average);
		//app_x << "------------average-------------" << endl;
		app_x << setw(3) << PhotoCount << setw(5) << map_feature.size();
		//app_x<<setw(15)<<ransac_match[u].match_set.size();
		app_x << setw(12) <<ransac_match[0].X[0] << setw(12) << ransac_match[0].X[1] << setw(12) <<ransac_match[0].X[2] << setw(12);
		app_x << ransac_match[0].X[3] * 180 / PI + 90 << setw(12);
		app_x <<ransac_match[0].X[4] * 180 / PI << setw(12);
		app_x << ransac_match[0].X[5] * 180 / PI << endl;

	}
 */	
	//app_x.close();

}