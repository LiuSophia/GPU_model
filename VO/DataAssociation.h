#ifndef _DataAssociation_
#define _DataAssociation_

#include <cv.h>

#include <vector>
#include <fstream>

#include "MapManagement.h"

using namespace std;

class DataAssociation
{

private:


public:


	//void Match_Old_Feature_Original( vector<SingleFeature>& feature, vector<Keep_Feature>& map_feature, vector<int>& find_map_feature, int& on_image_num, const int search_window_size, const double original_d_match_threshold, const int extended );
	//void Match_Old_Feature_Current( vector<SingleFeature>& feature, vector<Keep_Feature>& map_feature, vector<int>& find_map_feature, int& on_image_num, const int search_window_size, const double current_d_match_threshold, const int extended );
	
	void Search_Pair_Feature( const vector<SingleFeature> l_feature, vector<PairFeature>& pair_feature);


};



#endif
