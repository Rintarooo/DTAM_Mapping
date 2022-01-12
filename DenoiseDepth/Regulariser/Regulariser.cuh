#ifndef REGULARISER_CUH
#define REGULARISER_CUH 

// opencv
#include <opencv2/opencv.hpp>

// cuda
#include <opencv2/cudev/common.hpp>// CV_CUDEV_SAFE_CALL

void computeGCaller(const float* img, float* g,
				int width, int height, int pitch,
				float alphaG=3.5f, float betaG=1.0f);

void update_q_dCaller(const float *g, const float *a, 
				  float *q, float *d,
				  int width, int height,
				  float sigma_q, float sigma_d, float epsilon, float theta);

void minimizeEauxCaller(const float*cdata, int rows, int cols,
				 float*a, const float*d,
			 	const float*d_Cmin, const float*C_min, const float*C_max,
			 	float far, float near, int layers,
				 float theta, float lambda, float scale_Eaux, float*Eaux_);

#endif