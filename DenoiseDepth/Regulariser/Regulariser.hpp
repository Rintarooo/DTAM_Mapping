#ifndef REGULARISER_H
#define REGULARISER_H 

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>// https://docs.opencv.org/3.4.11/da/d17/group__ximgproc__filters.html#ga80b9b58fb85dd069691b709285ab985c
#include "vector_types.h"// cuda vector types, such as float3, float4


class Regulariser
{
public:	
	bool regularised_flag;
	cv::cuda::GpuMat q_, g_, a_, d_;
	// Regulariser() {};
	Regulariser(int rows, int cols, int layers, 
		float near, float far,
		float alphaG, float betaG, 
		float theta_start, float theta_min, 
		float epsilon, float lambda, float scale_Eaux);
	
	void compute_gWeight(const cv::cuda::GpuMat& referenceImageGray);
	void imwrite_gWeight() const;
	void optimize_global(int cnt,
		const cv::cuda::GpuMat &Cdata, 
		const cv::cuda::GpuMat &CminIdx, 
		const cv::cuda::GpuMat &Cmin,
		const cv::cuda::GpuMat &Cmax);
	void debug_cumat() const;
	void imwrite_depth_refined(int cnt) const;

private:
	const int rows, cols, layers;
	const float near, far;
	const float alphaG, betaG, theta_start, theta_min, epsilon;
	float lambda, scale_Eaux, sigma_d_, sigma_q_, depthStep;

	// for debugging
	cv::cuda::GpuMat Eaux_;

	void primal_dual_gHuber(
		const cv::cuda::GpuMat &Cdata, 
		const cv::cuda::GpuMat &CminIdx, 
		const cv::cuda::GpuMat &Cmin,
		const cv::cuda::GpuMat &Cmax);

	void computeSigmas(float theta);
	void update_q_d(float theta);
	void minimizeEaux(
		const cv::cuda::GpuMat &Cdata, 
		const cv::cuda::GpuMat &CminIdx, 
		const cv::cuda::GpuMat &Cmin,
		const cv::cuda::GpuMat &Cmax, 
		float theta);
	void plot_C_Eaux_Q(
		const cv::cuda::GpuMat &Cdata, 
		const cv::cuda::GpuMat &Cmax,
		int n);

}; 
#endif

