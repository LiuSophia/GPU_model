
#include "timer.h"
#include "main.h"
#include "kernelP3P.cuh"
using namespace std;
using namespace cv;
VO VO_P3P;

void InitialP3P()
{
	FILE *matchfeature =fopen("Paratxt//matchfeature1.txt","r");
	FILE *mapfeature =fopen("Paratxt//mapfeature1.txt","r");
	fstream para("Paratxt//parameter_DataSize1.txt",ios::in );
	fstream Camera( "Paratxt//Camera.txt", ios::in );
	int  col;
	char str[50];double value;
	
	
	
	while( para >> str >> value )
	{ 
		if ( !strcmp( str, "mapfeature_num" ) )			VO_P3P.mapNum						= value;
		else if ( !strcmp( str, "matchfeature_num" ) )		VO_P3P.featureNum				= value;
		else if ( !strcmp( str, "data_col" ) )              col					= value;
	}
	printf("mapfeature num: %d\n", VO_P3P.mapNum);
	printf("matchfeature num: %d\n", VO_P3P.featureNum);
	printf("parameter_size: %d\n", col);
	
	VO_P3P.match_feature.clear();
	VO_P3P.map_feature.clear();
	keepfeature temp,temp1;
	
	for(int i=0;i<VO_P3P.mapNum;i++)
	{
		memset(&temp1,0,sizeof(keepfeature));
		for(int j=0;j<col;j++)
		{
			if(j==0)	fscanf(mapfeature,"%d",&temp1.num);
			if(j==1)	fscanf(mapfeature,"%lf",&temp1.hx);
			if(j==2)	fscanf(mapfeature,"%lf",&temp1.hy);
			if(j==3)	fscanf(mapfeature,"%lf",&temp1.hz);
			if(j==4)	fscanf(mapfeature,"%lf",&temp1.l_ix);
			if(j==5)	fscanf(mapfeature,"%lf",&temp1.l_iy);
			if(j==6)	fscanf(mapfeature,"%lf",&temp1.depthvalue);
		}
		VO_P3P.map_feature.push_back(temp1);
	}
		
	for(int i=0;i<VO_P3P.featureNum;i++)
	{
		memset(&temp,0,sizeof(keepfeature));
		for(int j=0;j<col;j++)
		{
			if(j==0)	fscanf(matchfeature,"%d",&temp.num);
			if(j==1)	fscanf(matchfeature,"%lf",&temp.hx);
			if(j==2)	fscanf(matchfeature,"%lf",&temp.hy);
			if(j==3)	fscanf(matchfeature,"%lf",&temp.hz);
			if(j==4)	fscanf(matchfeature,"%lf",&temp.l_ix);
			if(j==5)	fscanf(matchfeature,"%lf",&temp.l_iy);
			if(j==6)	fscanf(matchfeature,"%lf",&temp.depthvalue);
		}
		VO_P3P.match_feature.push_back(temp);
	}
	
	
	while( Camera >> str >> value )
	{ 
		if ( !strcmp( str, "l_u0" ) )				VO_P3P.l_u0						= value;
		else if ( !strcmp( str, "l_v0" ) )			VO_P3P.l_v0						= value;
		else if ( !strcmp( str, "l_fu" ) )          VO_P3P.l_fu						= value;
		else if ( !strcmp( str, "l_fv" ) )          VO_P3P.l_fv						= value;
	}
	
	Ransac_Pos temp3;
	for (char w = 0; w < 3; w++)
	{
		for (char h = 0; h < 3; h++)
		{
			temp3.Rs[w][h] = 0;
		}
		temp3.X[w] = 0;
	}
	temp3.Rs[0][0] = 1;
	temp3.Rs[2][1] = -1;
	temp3.Rs[1][2] = 1;
	VO_P3P.VO_Pos.push_back(temp3);
	
	Camera.close();
	para.close();
	fclose(matchfeature);
	fclose(mapfeature);
	//cout<<l_u0<<setw(10)<<l_v0<<setw(10)<<l_fu<<setw(10)<<l_fv<<endl;
	
}



int main(int argc, char* argv[])
{

    //string fileId = std::to_string(3);
	InitialP3P();

	cout<<VO_P3P.match_feature.size()<<endl;
	cout<<VO_P3P.map_feature.size()<<endl;
	for(int i=0;i<VO_P3P.match_feature.size();i++)
	{
		cout<<VO_P3P.match_feature[i].num<<setw(10)<<VO_P3P.match_feature[i].hx<<setw(10)<<VO_P3P.match_feature[i].hy<<setw(10)<<VO_P3P.match_feature[i].hz<<setw(10)<<VO_P3P.match_feature[i].l_ix<<setw(10)<<VO_P3P.match_feature[i].l_iy<<setw(10)<<VO_P3P.match_feature[i].depthvalue<<endl;
	}
	
	for(int i=0;i<VO_P3P.map_feature.size();i++)
	{
		cout<<VO_P3P.map_feature[i].num<<setw(10)<<VO_P3P.map_feature[i].hx<<setw(10)<<VO_P3P.map_feature[i].hy<<setw(10)<<VO_P3P.map_feature[i].hz<<setw(10)<<VO_P3P.map_feature[i].l_ix<<setw(10)<<VO_P3P.map_feature[i].l_iy<<setw(10)<<VO_P3P.map_feature[i].depthvalue<<endl;
	}
	
	/* P3PSolver CPU_p3p;
	CPU_p3p.ransac_time=5;
	cout<<CPU_p3p.ransac_time<<endl; */
	int sol_num=(VO_P3P.match_feature.size()*(VO_P3P.match_feature.size()-1)*(VO_P3P.match_feature.size()-2))/6;//n取3
	double *CPU_WL,*CPU_Rs,*GPU_WL_dub,*GPU_Rs_dub;
	float *GPU_WL,*GPU_Rs;
	int *CPU_ransac_match, *GPU_ransac_match_dub, *GPU_ransac_match;
	CPU_WL=(double*)malloc(sol_num * 3*sizeof(double));
	GPU_WL=(float*)malloc(sol_num * 3*sizeof(float));
	CPU_Rs=(double*)malloc(sol_num * 9*sizeof(double));
	GPU_Rs=(float*)malloc(sol_num * 9*sizeof(float));
	CPU_ransac_match=(int*)malloc(sol_num *sizeof(int));
	GPU_ransac_match_dub=(int*)malloc(sol_num *sizeof(int));
	GPU_ransac_match=(int*)malloc(sol_num *sizeof(int));
	
	GPU_WL_dub=(double*)malloc(sol_num * 3*sizeof(double));
	GPU_Rs_dub=(double*)malloc(sol_num * 9*sizeof(double));
	
	VO_P3P.link_RANSAC.Run_RANSAC(VO_P3P.match_feature,VO_P3P.map_feature,VO_P3P.ransac_match,CPU_WL,CPU_Rs,CPU_ransac_match,VO_P3P.l_u0,VO_P3P.l_v0,VO_P3P.l_fu,VO_P3P.l_fv);
	VO_P3P.VO_Pos = VO_P3P.ransac_match;
	cout<<VO_P3P.ransac_match.size()<<endl;
	
	
	/* for(int i=0;i<3;i++)
			cout<<VO_P3P.VO_Pos[0].X[i]<<endl;
	
			cout<<VO_P3P.VO_Pos[0].X[3]* 180 / PI + 90<<endl;
			cout << VO_P3P.VO_Pos[0].X[4] * 180 / PI <<endl;
			cout << VO_P3P.VO_Pos[0].X[5] * 180 / PI << endl; */
	
	
	launchVOKernel(VO_P3P.match_feature,VO_P3P.map_feature,VO_P3P.ransac_match,GPU_WL,GPU_Rs,GPU_ransac_match,VO_P3P.l_u0,VO_P3P.l_v0,VO_P3P.l_fu,VO_P3P.l_fv);
	launchVOKernel_dub(VO_P3P.match_feature,VO_P3P.map_feature,VO_P3P.ransac_match,GPU_WL_dub,GPU_Rs_dub,GPU_ransac_match_dub,VO_P3P.l_u0,VO_P3P.l_v0,VO_P3P.l_fu,VO_P3P.l_fv);
	
	int maxsetsize=0,set=0;
	for(int i=0;i<sol_num;i++)
	{
		if(maxsetsize<CPU_ransac_match[i])
		{
			maxsetsize=CPU_ransac_match[i];
			set=i;
		}
			
	}
	cout<<"set"<<setw(5)<<set<<setw(10)<<"CPU maxsetsize "<<setw(10)<<maxsetsize<<endl;
	printf("CPU  WL: %.13f\t",CPU_WL[set*3]);
	printf("%.13f\t",CPU_WL[set*3+1]);
	printf("%.13f\n",CPU_WL[set*3+2]);
	printf("CPU Rs: %.13f\t",atan2(CPU_Rs[set*9+5], CPU_Rs[set*9+8])* 180 / PI + 90);
	printf("%.13f\t",asin(-CPU_Rs[set*9+2])* 180 / PI );
	printf("%.13f\n",atan2(CPU_Rs[set*9+1], CPU_Rs[set*9])* 180 / PI );
	cout<<"---------------------------------------------------"<<endl;
	struct timeval starttime, endtime;
	double executime;
	gettimeofday(&starttime, NULL);

	for(int i=0;i<sol_num;i++)
	{
		if(maxsetsize<GPU_ransac_match[i])
		{
			maxsetsize=GPU_ransac_match[i];
			set=i;
		}
			
	}
	
	gettimeofday(&endtime, NULL);
				
	executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
	executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
	printf("CPU find maxsetsize time: %13lf msec\n", executime);
	cout<<"---------------------------------------------------"<<endl;
	cout<<"set"<<setw(5)<<set<<setw(10)<<"GPU maxsetsize float"<<setw(10)<<maxsetsize<<endl;
	printf("GPU float WL: %.13f\t",GPU_WL[set*3]);
	printf("%.13f\t",GPU_WL[set*3+1]);
	printf("%.13f\n",GPU_WL[set*3+2]);
	printf("GPU float Rs: %.13f\t",atan2(GPU_Rs[set*9+5], GPU_Rs[set*9+8])* 180 / PI + 90);
	printf("%.13f\t",asin(-GPU_Rs[set*9+2])* 180 / PI );
	printf("%.13f\n",atan2(GPU_Rs[set*9+1], GPU_Rs[set*9])* 180 / PI );
	//cout<<"GPU_WL"<<setw(5)<<GPU_WL[set*3]<<setw(15)<<GPU_WL[set*3+1]<<setw(15)<<GPU_WL[set*3+2]<<endl;
	maxsetsize=0,set=0;
	for(int i=0;i<sol_num;i++)
	{
		if(maxsetsize<GPU_ransac_match_dub[i])
		{
			maxsetsize=GPU_ransac_match_dub[i];
			set=i;
		}
			
	}
	cout<<"---------------------------------------------------"<<endl;
	cout<<"set"<<setw(5)<<set<<setw(10)<<"GPU maxsetsize double"<<setw(10)<<maxsetsize<<endl;
	printf("GPU double WL: %.13f\t",GPU_WL_dub[set*3]);
	printf("%.13f\t",GPU_WL_dub[set*3+1]);
	printf("%.13f\n",GPU_WL_dub[set*3+2]);
	printf("GPU double Rs: %.13f\t",atan2(GPU_Rs_dub[set*9+5], GPU_Rs_dub[set*9+8])* 180 / PI + 90);
	printf("%.13f\t",asin(-GPU_Rs_dub[set*9+2])* 180 / PI );
	printf("%.13f\n",atan2(GPU_Rs_dub[set*9+1], GPU_Rs_dub[set*9])* 180 / PI );
	//cout<<"GPU_WL"<<setw(5)<<GPU_WL_dub[set*3]<<setw(15)<<GPU_WL_dub[set*3+1]<<setw(15)<<GPU_WL_dub[set*3+2]<<endl;
	cout<<"---------------------------------------------------"<<endl;
	
	
	
	int gpu_WL_f_t=0,gpu_Rs_f_t=0,gpu_WL_d_t=0,gpu_Rs_d_t=0,ransac_match_error=0;
	/* cout<<"-----------------------GPU sol WL float----------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		if(sqrt(CPU_WL[i*3]-GPU_WL[i*3])*(CPU_WL[i*3]-GPU_WL[i*3])>0.000000001)
		{
			gpu_WL_f_t++;
			cout<<i<<endl;
			cout<<"CPU_WL"<<setw(5)<<CPU_WL[i*3]<<setw(15)<<CPU_WL[i*3+1]<<setw(15)<<CPU_WL[i*3+2]<<endl;
			cout<<"GPU_WL"<<setw(5)<<GPU_WL[i*3]<<setw(15)<<GPU_WL[i*3+1]<<setw(15)<<GPU_WL[i*3+2]<<endl;
			cout<<endl;
		}
			
	}
	cout<<"GPU sol WL float   "<<gpu_WL_f_t<<endl;
	cout<<"-----------------------GPU sol WL float----------------------------"<<endl;
	cout<<"-----------------------GPU sol Rs float----------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		if(sqrt(CPU_Rs[i*3]-GPU_Rs[i*3])*(CPU_Rs[i*3]-GPU_Rs[i*3])>0.000000001)
		{
			gpu_Rs_f_t++;
			cout<<i<<endl;
			cout<<"CPU_Rs"<<setw(5)<<CPU_Rs[i*3]<<setw(15)<<CPU_Rs[i*3+1]<<setw(15)<<CPU_Rs[i*3+2]<<endl;
			cout<<"GPU_Rs"<<setw(5)<<GPU_Rs[i*3]<<setw(15)<<GPU_Rs[i*3+1]<<setw(15)<<GPU_Rs[i*3+2]<<endl;
			cout<<endl;
		}
			
	}
	cout<<"GPU sol Rs float   "<<gpu_Rs_f_t<<endl;
	cout<<"-----------------------GPU sol Rs float----------------------------"<<endl;
	
	 */
	
	//cout<<"------------------CPU GPU compare to sol ransac_match float-------------------"<<endl;
	/* for(int i=0;i<sol_num;i++)
	{
		if(CPU_ransac_match[i]!=GPU_ransac_match[i])
		{
			ransac_match_error++;
			cout<<i<<endl;
			cout<<"CPU_ransac_match"<<setw(5)<<CPU_ransac_match[i]<<setw(10)<<"GPU_ransac_match"<<setw(5)<<GPU_ransac_match[i]<<endl;
		}
	}
	if(ransac_match_error==0)	cout<<"CPU GPU float compare to sol ransac_match pass"<<endl; */
	//cout<<"-----------------CPU GPU compare to sol ransac_match float------------------"<<endl;
	
	ransac_match_error=0;
	//cout<<"------------------CPU GPU compare to sol ransac_match double----------------"<<endl;
	/* for(int i=0;i<sol_num;i++)
	{
		if(CPU_ransac_match[i]!=GPU_ransac_match_dub[i])
		{
			ransac_match_error++;
			cout<<i<<endl;
			cout<<"CPU_ransac_match"<<setw(5)<<CPU_ransac_match[i]<<setw(10)<<"GPU_ransac_match"<<setw(5)<<GPU_ransac_match_dub[i]<<endl;
		}
	}
	if(ransac_match_error==0)	cout<<"CPU GPU double compare to sol ransac_match pass"<<endl; */
	//cout<<"-----------------CPU GPU compare to sol ransac_match double---------------"<<endl;
	
	//cout<<"-----------------------GPU sol WL double----------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		if(sqrt(CPU_WL[i*3]-GPU_WL_dub[i*3])*(CPU_WL[i*3]-GPU_WL_dub[i*3])>0.00000000001)
		{
			gpu_WL_d_t++;
			cout<<i<<endl;
			cout<<"CPU_WL"<<setw(5)<<CPU_WL[i*3]<<setw(15)<<CPU_WL[i*3+1]<<setw(15)<<CPU_WL[i*3+2]<<endl;
			cout<<"GPU_WL"<<setw(5)<<GPU_WL_dub[i*3]<<setw(15)<<GPU_WL_dub[i*3+1]<<setw(15)<<GPU_WL_dub[i*3+2]<<endl;
			cout<<endl;
		}
			
	}
	if(gpu_WL_d_t==0)	cout<<"GPU sol WL double pass"<<endl;
	//cout<<"-----------------------GPU sol WL double----------------------------"<<endl;
	//cout<<"-----------------------GPU sol Rs double----------------------------"<<endl;
	for(int i=0;i<sol_num;i++)
	{
		if(sqrt(CPU_Rs[i*9]-GPU_Rs_dub[i*9])*(CPU_Rs[i*9]-GPU_Rs_dub[i*9])>0.000000001)
		{
			gpu_Rs_d_t++;
			cout<<i<<endl;
			cout<<"CPU_Rs"<<setw(5)<<CPU_Rs[i*3]<<setw(15)<<CPU_Rs[i*3+1]<<setw(15)<<CPU_Rs[i*3+2]<<endl;
			cout<<"GPU_Rs_dub"<<setw(5)<<GPU_Rs_dub[i*3]<<setw(15)<<GPU_Rs_dub[i*3+1]<<setw(15)<<GPU_Rs_dub[i*3+2]<<endl;
			cout<<endl;
		}
			
	}
	if(gpu_Rs_d_t==0)	cout<<"GPU sol Rs double pass"<<endl;
	//cout<<"-----------------------GPU sol Rs double----------------------------"<<endl;
	//waitKey(0);
	
	
    return 0;
}
