#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"// cuda vector types, such as float3, float4


class CostVolume
{
public:
	
	cv::cuda::GpuMat reference_image_color_, reference_image_gray_;
	cv::cuda::GpuMat current_image_color_, current_image_gray_;
	cv::cuda::GpuMat Cmin, Cmax, CminIdx, Cdata;
	
	// // CostVolume::gaussian_filter()// https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter6/07_gaussian_filter.cpp
	cv::cuda::GpuMat gaussian_cur_gray_, gaussian_ref_gray_;     
	// cv::Ptr<cv::cuda::Filter> filter3x3;

	int count_;
	 
	// CostVolume() {};
	CostVolume(int _rows, int _cols, int _layers, float _near, float _far, const cv::Mat& Kcpu, std::string similarity);
	~CostVolume() {};
  
	void set_refRGB(const cv::Mat& reference_image, const cv::Mat& Rrw, const cv::Mat& trw);
	void update_CostVolume(const cv::Mat& image, const cv::Mat& Rmw, const cv::Mat& tmw);
	void getCW_fromWC(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&) const;
	void imwrite_inv_depth() const;
	void imwrite_RGB(const cv::Mat&) const;
	void plot_Cdata() const;
	void debug_Cdata() const;
	void debug_projection() const;
	void reset();

private:
	const int rows, cols, layers;
	const float near, far;
	const cv::Mat K_cpu;
	const std::string similarity;

	float depthStep;
	cv::Mat K_cpu_inv;
	cv::cuda::GpuMat K_gpu, K_gpu_inv;

	cv::Mat Twr, Tmr;
	cv::cuda::GpuMat Tmr_gpu;
	float Cdata_initv;	
};

#endif
