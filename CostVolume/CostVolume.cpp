#include "CostVolume.hpp"
#include "CostVolume.cuh"

CostVolume::CostVolume(int _rows, int _cols, int _layers, float _near, float _far, const cv::Mat& K_cpu, std::string similarity):
	rows(_rows), cols(_cols), layers(_layers), near(_near), far(_far), K_cpu(K_cpu), similarity(similarity)
{
	CV_Assert(layers >= 8);
	CV_Assert(near > far);
	depthStep = (near - far) / (layers - 1);
	cv::cuda::createContinuous(3, 3, CV_32FC1, K_gpu);// allocate continuous GpuMat data 
	K_gpu.upload(K_cpu);	
	/* debug
	for(int l=layers_-1; l >= 0; l--) {
		const float d_ = far_ + float(l)*depthStep;
		if(l <= 1 || layers_-2 <= l) std::cout << "index: " << l << ", inv_depth: " << d_ << std::endl;

	float* ptrK = (float*)K_cpu.data;// Mat.data returns uchar*, so cast into float*
	for (int i = 0; i < K_cpu.total(); i++){
		std::cout << ptrK[i] << std::endl;
	}
	cv::Mat K_cpu = (cv::Mat_<float>(3,3) << fx, 0.0,  cx,
											0.0,  fy,  cy,
											0.0, 0.0, 1.0);
	cv::Mat K_cpu_inv = (cv::Mat_<float>(3,3) << 1/fx,  0.0, -cx/fx,
												  0.0, 1/fy, -cy/fy,
												  0.0,  0.0,    1.0);

	CV_Assert(image.isContinuous()); -> https://koshinran.hateblo.jp/entry/2018/11/29/182224
	*/
	K_cpu_inv = K_cpu.inv();
	cv::cuda::createContinuous(3, 3, CV_32FC1, K_gpu_inv);
	K_gpu_inv.upload(K_cpu_inv);
	count_ = 0;
	cv::cuda::createContinuous(rows, cols, CV_32FC1, Cmin);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, Cmax);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, CminIdx);
	
	const cv::Mat CminIdx_cpu = far * cv::Mat::ones(rows, cols, CV_32FC1);
	CminIdx.upload(CminIdx_cpu);
	
	cv::cuda::createContinuous(layers, rows*cols, CV_32FC1, Cdata);
	Cdata_initv = 256.0f;//1e+30;//FLT_MAX;//1e+10;
	const cv::Mat Cdata_cpu = Cdata_initv * cv::Mat::ones(layers, rows*cols, CV_32FC1);
	Cdata.upload(Cdata_cpu);
	
	cv::cuda::createContinuous(rows, cols, CV_32FC4, reference_image_color_);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, reference_image_gray_);
	cv::cuda::createContinuous(rows, cols, CV_32FC4, current_image_color_);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, current_image_gray_);
	Twr.create(4, 4, CV_32FC1);
	Tmr.create(4, 4, CV_32FC1);
	cv::cuda::createContinuous(4, 4, CV_32FC1, Tmr_gpu);	
	
	// filter3x3 = cv::cuda::createGaussianFilter(CV_32FC1,CV_32FC1,cv::Size(3,3),0.3);//sigma=0.3
	cv::cuda::createContinuous(rows, cols, CV_32FC1, gaussian_cur_gray_);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, gaussian_ref_gray_);
}

void CostVolume::set_refRGB(const cv::Mat& img, const cv::Mat& Rwr, const cv::Mat& twr)
{
	CV_Assert(img.type() == CV_32FC4 && img.isContinuous());
	CV_Assert(img.rows % 32 == 0 && img.cols % 32 == 0 && img.cols >= 64);
	reference_image_color_.upload(img);
	cv::cuda::cvtColor(reference_image_color_, reference_image_gray_, cv::COLOR_BGRA2GRAY); // conversion on gpu presumably faster
	Twr = (cv::Mat_<float>(4,4) <<
		Rwr.at<float>(0,0), Rwr.at<float>(0,1), Rwr.at<float>(0,2), twr.at<float>(0),
		Rwr.at<float>(1,0), Rwr.at<float>(1,1), Rwr.at<float>(1,2), twr.at<float>(1),
		Rwr.at<float>(2,0), Rwr.at<float>(2,1), Rwr.at<float>(2,2), twr.at<float>(2),
		0,					0,					 0,				      1);
	// filter3x3->apply(reference_image_gray_, gaussian_ref_gray_);		
}

void CostVolume::update_CostVolume(const cv::Mat& img, const cv::Mat& Rwm, const cv::Mat& twm)
{
	CV_Assert(img.type() == CV_32FC4 && img.isContinuous());
	current_image_color_.upload(img);
	cv::cuda::cvtColor(current_image_color_, current_image_gray_, cv::COLOR_BGRA2GRAY); // conversion on gpu presumably faster

	cv::Mat Rmw, tmw, Tmw;// from (w)orld to (m)
	Rmw.create(3, 3, CV_32FC1);	
	tmw.create(3, 1, CV_32FC1);	
	this->getCW_fromWC(Rwm, twm, Rmw, tmw);
	Tmw.create(4, 4, CV_32FC1);	
	Tmw = (cv::Mat_<float>(4,4) <<
		Rmw.at<float>(0,0), Rmw.at<float>(0,1), Rmw.at<float>(0,2), tmw.at<float>(0),
		Rmw.at<float>(1,0), Rmw.at<float>(1,1), Rmw.at<float>(1,2), tmw.at<float>(1),
		Rmw.at<float>(2,0), Rmw.at<float>(2,1), Rmw.at<float>(2,2), tmw.at<float>(2),
		0,					0,					 0,				      1);

	Tmr = Tmw * Twr;// Tmr = Tmw * Trw.inv();
	Tmr_gpu.upload(Tmr);

	// debug_projection();
	// filter3x3->apply(current_image_gray_, gaussian_cur_gray_);
	updateCostVolumeCaller( (float*)K_gpu.data, (float*)K_gpu_inv.data, (float*)Tmr_gpu.data,
							rows, cols, current_image_color_.step,
							near, far, layers, rows*cols,
							(float*)(Cdata.data), (float)count_,
							(float*)(Cmin.data), (float*)(Cmax.data), (float*)(CminIdx.data),
							(float*)(reference_image_gray_.data), (float*)(current_image_gray_.data),
							(float*)(gaussian_ref_gray_.data), (float*)(gaussian_cur_gray_.data),
							(float4*)(reference_image_color_.data), (float4*)(current_image_color_.data),
							similarity);
	
}

void CostVolume::getCW_fromWC(const cv::Mat &Rwc, const cv::Mat &twc, cv::Mat &Rcw, cv::Mat &tcw) const{
	/*
	Rwc is rotation matrix from (c)amera frame to (w)orld frame.
	twc is translation vector from (c)amera frame to (w)orld frame.
	
	Rcw = Rwc^-1 = Rwc.t();
	P = K[Rcw|tcw] = K[Rwc^(-1)|-Rwc^(-1)*twc] = K[Rwc^t|-Rwc^t*twc]
	https://www.slideshare.net/SSII_Slides/ssii2019ts3-149136612/28
	*/
	Rcw =  Rwc.t();
	tcw = -Rwc.t()*twc;
}


void CostVolume::imwrite_inv_depth() const{

	cv::Mat src, dst;
	src.create(rows, cols, CV_32FC1);     
	dst.create(rows, cols, CV_8UC1);     
	CV_Assert(CminIdx.type() == CV_32FC1);
	CminIdx.download(src);	
	// cv::normalize(src, dst, 0, 255, CV_MINMAX, CV_8U);
	//// scaling each float value[minv, maxv] into 8bit unsigned int value[0,255] 
	const float minv = far, maxv = near;// minv = 1./14., maxv = 1./4.;
	for(int v=0; v <src.rows; v++) {
	   const float* src_ptr = src.ptr<float>(v); 
	   std::uint8_t* dst_ptr = dst.ptr<std::uint8_t>(v);
	   for(int u=0; u<src.cols; u++) {
			dst_ptr[u] = static_cast<std::uint8_t>((src_ptr[u] - minv) * 255. /(maxv - minv));
	   }
	}

	std::string img_filename = "inv_depth" + std::to_string(this->count_) + ".png";
	cv::imwrite(img_filename, dst);
}

void CostVolume::imwrite_RGB(const cv::Mat& img) const{
	/*
	cv::cuda::GpuMat cur_rgb_gpu;
	cv::cuda::createContinuous(rows, cols, CV_32FC3, cur_rgb_gpu);
	cv::cuda::cvtColor(current_image_color_, cur_rgb_gpu, cv::COLOR_BGRA2BGR);
	cv::Mat cur_rgb;
	cur_rgb_gpu.download(cur_rgb);

	cv::Mat cur_gray;
	current_image_gray_.download(cur_gray);
	cv::imwrite("cur_gray" + std::to_string(this->count_) + ".png", cur_gray);
	*/
	cv::Mat dst;
	CV_Assert(img.type() == CV_32FC4);
	cvtColor(img, dst, CV_BGRA2BGR);
	dst.convertTo(dst, CV_8UC3);
	const std::string extension = ".png"; 
	std::string img_filename = "cur_rgb" + std::to_string(this->count_) + extension; 
	if(this->count_ == 0) img_filename = "cur_rgb" + std::to_string(this->count_+1) + "_ref" + extension; 
	cv::imwrite(img_filename, dst);
}

void CostVolume::plot_Cdata() const{
	FILE *gid;
	const char* gnuplot_path = "/usr/bin/gnuplot"; 
	gid = popen(gnuplot_path, "w");
	if (gid == NULL) {
		std::cerr << "check gnuplot path '/usr/bin/gnuplot'" << std::endl;
		std::exit(1);
	}
	cv::Mat C_float;
	Cdata.download(C_float);
	float* C_float_ptr = (float*)C_float.data;
	// for(int y = 0; y < rows; y++){
	// 	for(int x = 0; x < cols; x++){
	int x = 224, y = 217;
	fprintf(gid, "set terminal png\n");// save .png
	fprintf(gid, "set output 'graph_x%d_y%d.png'\n", x, y);
	fprintf(gid, "plot '-' with lines\n");	
			int i = x + y*cols;
			for(int l=layers-1; l >= 0; l--) {
				const float d = far + float(l)*depthStep;// d is inverse depth
				const float CostVolume_v = C_float_ptr[i+l*rows*cols];
				fprintf(gid, "%f %f\n", d, CostVolume_v);
				// fprintf(gid, "plot './uv.txt' with dots");
			}
			fprintf(gid, "e\n");// end
	// 	}
	// }
	pclose(gid);
}


void CostVolume::debug_Cdata() const{
	cv::Mat Cmin_float, CminIdx_float, Cmax_float, C_float;
	CminIdx.download(CminIdx_float);
	Cmin.download(Cmin_float);
	Cmax.download(Cmax_float);
	Cdata.download(C_float);
	
	float* C_float_ptr = (float*)C_float.data;
	int x = 440, y = 300;
	int i = x + y*cols;
	for(int l=layers-1; l >= 0; l--) {
		printf("CostVolume voxel[%d][%d][%d]: %f, inv_depth: %f\n", x, y, l, C_float_ptr[i+l*rows*cols], far+depthStep*l);
	}

	// printf("Cdata[0]: %f\n", C_float_ptr[0]);
	// printf("Cdata[1]: %f\n", C_float_ptr[1]);
	// printf("Cdata[2]: %f\n", C_float_ptr[2]);
	float* Cmax_ptr = (float*)Cmax_float.data;
	printf("Cmax[i]: %f\n", Cmax_ptr[i]);
	// printf("Cmax[0]: %f\n", Cmax_ptr[0]);
	// printf("Cmax[1]: %f\n", Cmax_ptr[1]);
	// printf("Cmax[2]: %f\n", Cmax_ptr[2]);
	float* Cmin_ptr = (float*)Cmin_float.data;
	printf("Cmin[i]: %f\n", Cmin_ptr[i]);
	// printf("Cmin[0]: %f\n", Cmin_ptr[0]);
	// printf("Cmin[1]: %f\n", Cmin_ptr[1]);
	// printf("Cmin[2]: %f\n", Cmin_ptr[2]);
	float* CminIdx_ptr = (float*)CminIdx_float.data;
	printf("CminIdx[i]: %f\n", CminIdx_ptr[i]);
	// printf("CminIdx[0]: %f\n", CminIdx_ptr[0]);
	// printf("CminIdx[1]: %f\n", CminIdx_ptr[1]);
	// printf("CminIdx[2]: %f\n", CminIdx_ptr[2]);
	
	/*
	// float* Cmin_ptr = (float*)Cmin_float.data;
	float* CminIdx_ptr = (float*)CminIdx_float.data;
	// float* Cmax_ptr = (float*)Cmax_float.data;
	// std::cout << Cmin_ptr[0] << std::endl;
	std::cout << std::to_string(this->count_) << std::endl;
	// for(int y = 0; y < rows; y++){
		// for(int x = 0; x < cols; x++){
		for(int x = 270; x < 290; x++){
			// int i = x + y*cols;
			int i = x + 240*cols;
			// printf("Cmax_ptr[%d]: %f\n", i, Cmax_ptr[i]);
			// printf("Cmin_ptr[%d]: %f\n", i, Cmin_ptr[i]);
			
			printf("CminIdx_ptr(=inv depth)[%d]: %f\n", i, CminIdx_ptr[i]);// =depth
		}*/
	// }

}


void CostVolume::debug_projection() const{
	float ur, vr;
	ur = 440, vr = 300;
	float* K_cpu_ptr = (float*)K_cpu.data;
	float* K_cpu_inv_ptr = (float*)K_cpu_inv.data;
	std::cout << "K:" << std::endl;
	std::cout << K_cpu_ptr[0] <<  std::endl;
	std::cout << K_cpu_ptr[2] <<  std::endl;
	std::cout << K_cpu_ptr[4] <<  std::endl;
	std::cout << K_cpu_ptr[5] <<  std::endl;
	std::cout << "K inv:" << std::endl;
	// float* Twr_ptr = (float*)Trw.inv().data;
	float* Tmr_ptr = (float*)Tmr.data;
	std::cout << "Tmr: " << Tmr <<  std::endl;
	std::cout << Tmr_ptr[5] <<  std::endl;
	
	for(int l=layers-1; l >= 0; l--) {
		float d = far + float(l)*depthStep;// d is inverse depth
		float zr = 1.0/d; // zr is depth, divide by 0 is evaluated as Inf, as per IEEE-754
		float xr = (K_cpu_inv_ptr[0]*ur + K_cpu_inv_ptr[2])*zr;
		float yr = (K_cpu_inv_ptr[4]*vr + K_cpu_inv_ptr[5])*zr;
		// std::cout << "xr: " << xr << ", yr: " << yr << ", zr: " << zr << std::endl;		
		float xm = Tmr_ptr[0]*xr + Tmr_ptr[1]*yr + Tmr_ptr[2]*zr  + Tmr_ptr[3];
		float ym = Tmr_ptr[4]*xr + Tmr_ptr[5]*yr + Tmr_ptr[6]*zr  + Tmr_ptr[7];
		float zm = Tmr_ptr[8]*xr + Tmr_ptr[9]*yr + Tmr_ptr[10]*zr + Tmr_ptr[11];
		float um = K_cpu_ptr[0]*(xm/zm) + K_cpu_ptr[2];
		float vm = K_cpu_ptr[4]*(ym/zm) + K_cpu_ptr[5];
		std::cout << "inv d: " << d << ", um: " << um << ", vm: " << vm << std::endl;
	}
	std::cout << "ur: " << ur << ", vr: " << vr << std::endl << std::endl;
}

void CostVolume::reset()
{
	count_	  = 0;
	const cv::Mat Cmin_cpu = Cdata_initv * cv::Mat::ones(rows, cols, CV_32FC1);
	Cmin.upload(Cmin_cpu);
	const cv::Mat CminIdx_cpu = far * cv::Mat::ones(rows, cols, CV_32FC1);
	CminIdx.upload(CminIdx_cpu);
	const cv::Mat Cmax_cpu = cv::Mat::zeros(rows, cols, CV_32FC1);
	Cmax.upload(Cmax_cpu);
	const cv::Mat Cdata_cpu = Cdata_initv * cv::Mat::ones(layers, rows*cols, CV_32FC1);
	Cdata.upload(Cdata_cpu);
}

