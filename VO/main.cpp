#include "main.h"
using namespace std;
using namespace cv;

SHLiuDlg Run_program;

void SHLiuDlg::LoadImage()
{

	char path[100];
	
	snprintf(path, sizeof(path),"fig//rgb//%d.png", PhotoCount);

	ifstream in_file(path, ios::in);

	if (in_file)
	{
		//Image = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);
		Image = imread(path, IMREAD_GRAYSCALE);
	}
	else
	{
		in_image0 = true;
	}

	in_file.close();

}

void SHLiuDlg::WORK( const Mat l_image, const double SampleTime)
{
	struct timeval starttime, endtime;
	double executime;
	if(Run_program.PhotoCount==5)
		gettimeofday(&starttime, NULL);
	
	Run_program.BinocularVO.Run_VO(l_image,SampleTime,Run_program.PhotoCount);
	
	if(Run_program.PhotoCount==5)
	{
		gettimeofday(&endtime, NULL);
				
		executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
		executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
		printf("Overall time: %13lf msec\n", executime);
	}
	
}

/*void CALLBACK TimeProc(UINT uTimerID, UINT uMsg, DWORD dwUser, DWORD dw1, DWORD dw2)
{
	SHLiuDlg *pointer = (SHLiuDlg *)dwUser;

	pointer->DoEvent(); //要重複執行的函式
}*/

void SHLiuDlg::DoEvent()
{
	time1 = (double)cvGetTickCount(); //***********************************************************************************


	//SampleTime = SampleTime_temp[PhotoCount];
		
	//time3 = (double)cvGetTickCount(); //***********************************************************************************		

				//ftpclass.FTPDownload(PhotoCount);

	Run_program.LoadImage();
		
	//time4 = (double)cvGetTickCount(); //***********************************************************************************

	if (in_image0 == true /*&& in_image1 == true /*&& in_image2==true && in_image3 == true*/)
	{
		PhotoCount -= 1;
		cout<<"圖片讀取結束!"<<endl;

	}
	else
	{
		Run_program.WORK(Image,SampleTime);
		PhotoCount++;
	}
	
	time2 = (double)cvGetTickCount(); //***********************************************************************************

		//m_Time1 = (int)( (time2 - time1)/(cvGetTickFrequency()*1000.) ); // 系統全部
		//m_Frequency2 = cvRound(1./(((time2 - time1)/(cvGetTickFrequency()*1000.))/1000.)); // 系統全部 頻率

}

int main(int argc, char* argv[])
{
	
	Run_program.PhotoCount=0;
	Run_program.BinocularVO.Initial_VO();
	while(Run_program.in_image0== false)
	{
		Run_program.DoEvent();
	}
	
	
	/*UINT uDelay = 1;//m_SampleTime 為自訂的取樣時間 單位:毫秒
	UINT uResolution = 1;
	DWORD dwUser = (DWORD)this;
	UINT fuEvent = TIME_PERIODIC; //You also choose TIME_ONESHOT;

	timeBeginPeriod(1); //精度1ms
	FTimerID = timeSetEvent(uDelay, uResolution, TimeProc, dwUser, fuEvent);*/
	
}