#include "DataAssociation.h"

void DataAssociation::Search_Pair_Feature( const vector<SingleFeature> l_feature, vector<PairFeature>& pair_feature)
{

	PairFeature temp;

	for( int i=0 ; i < (int)l_feature.size() ; i++ )
	{
			memset( &temp, 0, sizeof(PairFeature) ); // temp 初始化為零

			temp.ix	       = l_feature[i].ix;
			temp.iy        = l_feature[i].iy;			
			temp.laplacian = l_feature[i].laplacian;
			temp.size	   = l_feature[i].size;
			temp.dir	   = l_feature[i].dir;
			temp.hessian   = l_feature[i].hessian;
			memcpy( temp.descriptor, l_feature[i].descriptor, sizeof(temp.descriptor) ); // 複製向量
			temp.depthvalue= l_feature[i].depthvalue;
		
			temp.on_map	   = l_feature[i].on_map;

			pair_feature.push_back(temp);

	}

}