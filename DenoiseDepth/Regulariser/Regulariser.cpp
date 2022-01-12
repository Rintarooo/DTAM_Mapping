#include "Regulariser.hpp"
#include "Regulariser.cuh"

Regulariser::Regulariser(int rows, int cols, int layers, 
	float near, float far,
	float alphaG, float betaG, 
	float theta_start, float theta_min, 
	float epsilon, float lambda, float scale_Eaux) :
	rows(rows), cols(cols), layers(layers),
	near(near), far(far),
	alphaG(alphaG), betaG(betaG),
	theta_start(theta_start), theta_min(theta_min),
	epsilon(epsilon), lambda(lambda), scale_Eaux(scale_Eaux)
{
	// allocate g_ and q_
	cv::cuda::createContinuous(  rows, cols, CV_32FC1, g_);
	cv::cuda::createContinuous(2*rows, cols, CV_32FC1, q_);
	cv::cuda::createContinuous(  rows, cols, CV_32FC1, a_);
	cv::cuda::createContinuous(  rows, cols, CV_32FC1, d_);
	// for debugging
	cv::cuda::createContinuous(rows*cols, layers, CV_32FC1, Eaux_);
	
	const cv::Mat zero = cv::Mat::zeros(rows*cols, layers, CV_32FC1);
	Eaux_.upload(zero);
	depthStep = (near - far) / (layers - 1);
	
	regularised_flag = false;
	// CV_Assert(g_.step == static_cast<std::uint8_t>(cols*4) && q_.step == static_cast<std::uint8_t>(cols*4));
	CV_Assert(static_cast<int>(g_.step) == cols*4 && static_cast<int>(q_.step) == cols*4);
	CV_Assert(near > far);
	CV_Assert(layers >= 8);
	CV_Assert(theta_start > theta_min);
}

void Regulariser::compute_gWeight(const cv::cuda::GpuMat& referenceImageGray)
{
	/*
	cv::cuda::GpuMat ref_rgb_gpu;
	// cv::cuda::createContinuous(rows, cols, CV_32FC4, reference_image_color_);
	cv::cuda::createContinuous(rows, cols, CV_32FC3, ref_rgb_gpu);
	// cv::cuda::cvtColor(referenceImageGray, reference_image_color_, cv::COLOR_GRAY2BGRA);
	cv::cuda::cvtColor(referenceImageGray, ref_rgb_gpu, cv::COLOR_GRAY2BGR);
	ref_rgb_gpu.download(ref_rgb);
	// cvtColor(ref_rgb, ref_rgb, CV_BGRA2BGR);
	CV_Assert(ref_rgb.type() == CV_32FC3);
	*/		
	
	// Call the gpu function for caching g's
	computeGCaller( (float*)referenceImageGray.data,
					(float*)g_.data,
					referenceImageGray.cols, referenceImageGray.rows, referenceImageGray.step, 
					alphaG, betaG);
}

void Regulariser::imwrite_gWeight() const
{
	// Newcombe Phd paper, p.105/324, Figure 4.11(b)Weighting function
	/*cv::Mat gImg;//, dst;
	gImg.create(rows, cols, CV_32FC1);
	g_.download(gImg);
	cv::normalize(gImg, gImg, 0, 255, CV_MINMAX, CV_8U);
	cv::imwrite("gweight.png", gImg);*/

	/*
	//// *debug*
	float* g_ptr = (float*)gImg.data;
	// float* g_ptr = (float*)dst.data;
	for (int i = 0; i < gImg.total(); i++){
		std::cout << g_ptr[i] << std::endl;
   }
   */
	
	// cv::imshow("g", gImg);
	// cv::waitKey(10);
	
	cv::Mat src, dst;
	src.create(rows, cols, CV_32FC1);
	dst.create(rows, cols, CV_8UC1);     
	CV_Assert(g_.type() == CV_32FC1);
	g_.download(src);
	const float minv = 0, maxv = 1;// 0 < g < 1, because g = e^{-x}, 0 < x
	for(int v = 0; v < rows; v++) {
		const float* src_ptr = src.ptr<float>(v); 
		std::uint8_t* dst_ptr = dst.ptr<std::uint8_t>(v);
		for(int u = 0; u < cols; u++) {
			dst_ptr[u] = static_cast<std::uint8_t>((src_ptr[u] - minv) * 255. /(maxv - minv));
		}
	}
	const std::string png = "gweight.png";
	cv::imwrite(png, dst);
}


void Regulariser::optimize_global(
	int cnt, 
	const cv::cuda::GpuMat &Cdata, 
	const cv::cuda::GpuMat &CminIdx, 
	const cv::cuda::GpuMat &Cmin,
	const cv::cuda::GpuMat &Cmax)
{
	this->primal_dual_gHuber(Cdata, CminIdx, Cmin, Cmax);
	regularised_flag = true;				
}

void Regulariser::primal_dual_gHuber(
	const cv::cuda::GpuMat &Cdata, 
	const cv::cuda::GpuMat &CminIdx, 
	const cv::cuda::GpuMat &Cmin,
	const cv::cuda::GpuMat &Cmax)
{
	const cv::Mat zeroMat = cv::Mat::zeros(2*rows, cols, CV_32FC1);
	q_.upload(zeroMat);// initialize q_ with zero 
	CminIdx.copyTo(d_);// initialize d_ and a_ with the data cost minimum
	CminIdx.copyTo(a_);
	
	int n = 1;
	float theta = theta_start;
	while(theta > theta_min) {
		this->computeSigmas(theta);
		for (int i = 0; i < 10; i++) this->update_q_d(theta);
		this->minimizeEaux(Cdata, CminIdx, Cmin, Cmax, theta);
		
		const float beta = (theta >= 1e-3)? 1e-3 : 1e-4;// phd thesis p.140/324
		theta *= (1-beta*n);
		if(n == 4 || n == 20 || n == 40 || n == 60) {
			// std::cout << "sigma_d_: " << sigma_d_ << std::endl;
			// std::cout << "sigma_q_: " << sigma_q_ << std::endl;
			std::cout << "plot C, Eaux and Q" << std::endl;
			this->plot_C_Eaux_Q(Cdata, Cmax, n);
		}
		n++; std::cout << "iter: " << n << std::endl;// " theta: " << theta << 
		// debug
		// if(n == 4 || n == 20 || n == 40 || n == 60 || n == 80 || n == 120 || n == 160 || n == 200) this->imwrite_depth_refined(n);
	}
}

void Regulariser::update_q_d(float theta)
{
	update_q_dCaller((float*)g_.data, (float*)a_.data,
					 (float*)q_.data, (float*)d_.data,
					 a_.cols, a_.rows,
					 sigma_q_, sigma_d_, epsilon, theta);
}

void Regulariser::minimizeEaux(
	const cv::cuda::GpuMat &Cdata, 
	const cv::cuda::GpuMat &CminIdx, 
	const cv::cuda::GpuMat &Cmin,
	const cv::cuda::GpuMat &Cmax, 
	float theta)
{
	minimizeEauxCaller((float*)Cdata.data, rows, cols,
			(float*)a_.data, (float*)d_.data,
			(float*)CminIdx.data, (float*)Cmin.data, (float*)Cmax.data,
			far, near, layers,
			theta, lambda, scale_Eaux, (float*)Eaux_.data);
}

void Regulariser::computeSigmas(float theta)
{
/*
	The DTAM paper only provides a reference [3] for setting sigma_q & sigma_d
	[3] A. Chambolle and T. Pock. A first-order primal-dual
	algorithm for convex problems with applications to imaging.
	Journal of Mathematical Imaging and Vision, 40(1):120-
	145, 2011.
	https://hal.archives-ouvertes.fr/hal-00490826/document
	The relevant section of this (equation) dense paper is:
	p.30,31/50, Sec. 6.2.3 The Huber-ROF Model, ALG3
	
	compare p.31/50 equ(71) and DTAM paper equ(10)
	\gamma = \lambda = 1/\theta in DTAM paper
	\delta = \alpha = \epsilon i.e., Huber epsilon in DTAM paper
	\mu = 2*\sqrt{\gamma*\delta}/L
	L is defined in Theorem 1 in p.5/50 as L = ||K||, and ||K|| is defined in Sec. 2., Eqn. 1 in p.3/50 as:
	||K|| = max {Kx : x in X with ||x|| <= 1}.
	In our case, working from Sec. 2., eqn. 3 in p.3/50(see also Sec. 6.2.1, eqn. 63 in p.25/50 on how eqn. 3 is mapped to the ROF model),
	K is the forward differentiation matrix with G weighting, (AG in the paper), so ||K|| = 2,
	obtained for x = (0.5, -0.5, 0.5, -0.5, 0, 0, ..., 0). 
*/

// https://github.com/anuranbaka/OpenDTAM/blob/2.4.9_experimental/Cpp/DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.cpp#L135
	float L = 4.0;//2.0;

	float mu = 2.0*std::sqrt(epsilon/theta)/L;

	// TODO: check the original paper for correctness of these settings
	// p.19/50 equ(49) and p.21/50 Algorithm3 equ(60)
	sigma_d_ = mu/(2.0/theta);
	sigma_q_ = mu/(2.0*epsilon);
}

void Regulariser::imwrite_depth_refined(int cnt) const
{
	const std::string filename_d = "inv_depth" + std::to_string(cnt) + "_regu_d.png";
	const std::string filename_a = "inv_depth" + std::to_string(cnt) + "_regu_a.png";
	cv::Mat src_d, src_a, dst_d, dst_a;
	src_d.create(rows, cols, CV_32FC1);     
	src_a.create(rows, cols, CV_32FC1);     
	dst_d.create(rows, cols, CV_8UC1);     
	dst_a.create(rows, cols, CV_8UC1);	
	CV_Assert(d_.type() == CV_32FC1 && d_.isContinuous());
	CV_Assert(a_.type() == CV_32FC1 && a_.isContinuous());
	d_.download(src_d);
	a_.download(src_a);
	// cv::normalize(d_cpu, d_cpu, 0, 255, CV_MINMAX, CV_8U);
	const float minv = far, maxv = near;// minv = 1./14., maxv = 1./4.;
	for(int v=0; v < rows; v++) {
		const float* src_ptr_d = src_d.ptr<float>(v); 
		const float* src_ptr_a = src_a.ptr<float>(v); 
		std::uint8_t* dst_ptr_d = dst_d.ptr<std::uint8_t>(v);
		std::uint8_t* dst_ptr_a = dst_a.ptr<std::uint8_t>(v);
		for(int u=0; u< cols; u++) {
			dst_ptr_d[u] = static_cast<std::uint8_t>((src_ptr_d[u] - minv) * 255. /(maxv - minv));
			dst_ptr_a[u] = static_cast<std::uint8_t>((src_ptr_a[u] - minv) * 255. /(maxv - minv));
		}
	}

	cv::imwrite(filename_d, dst_d);
	cv::imwrite(filename_a, dst_a);
}


void Regulariser::debug_cumat() const{
	// for debug
	cv::Mat q_mat;
	q_.download(q_mat);
	
	float* q_mat_ptr = (float*)q_mat.data;
	// int x = 440;//25;
	// int y = 300;//260;
	for(int x=300; x<330; x++){
		for(int y=280; y<300; y++){
			int i = x + y*cols;
			printf("q_mat_ptr[%d][%d]: %f\n", x, y, q_mat_ptr[i]);
		}
	}
}

void Regulariser::plot_C_Eaux_Q(
	const cv::cuda::GpuMat &Cdata, 
	const cv::cuda::GpuMat &Cmax, 
	int n)
{
	FILE *gid;
	const char* gnuplot_path = "/usr/bin/gnuplot"; 
	gid = popen(gnuplot_path, "w");
	if (gid == NULL) {
		std::cerr << "check gnuplot path";
		std::exit(1);
	}
	cv::Mat C_cpu, Cmax_cpu;
	Cdata.download(C_cpu);
	Cmax.download(Cmax_cpu);
	float* C_ptr = (float*)C_cpu.data;
	float* Cmax_ptr = (float*)Cmax_cpu.data;
	// for(int y = 0; y < rows; y++){
	// 	for(int x = 0; x < cols; x++){
	int x = 295, y = 205;//int x = 224, y = 217;
	int i = x + y*cols;
	fprintf(gid, "set terminal png\n");
	fprintf(gid, "set output 'graph_x%d_y%d_n%d.png'\n", x, y, n);
	fprintf(gid, "set title 'DTAM paper Figure.4'\n");
	fprintf(gid, "set yrange[0:%d]\n", int(Cmax_ptr[i]+50.));
	fprintf(gid, "set xlabel 'inverse depth'\n");
	fprintf(gid, "set ylabel 'photomeric error'\n");
	// fprintf(gid, "plot '-' with lines\n");
	// fprintf(gid, "plot '-' with lines lt rgb 'red', '-' with lines lt rgb 'green'\n");// lt = linetype
	fprintf(gid, "plot '-' with lines lw 3 lt rgb 'red' title 'C(u)', '-' with lines dt 2 lw 2 lc rgb 'green' title 'Eaux(u)', '-' with lines lw 2 lt rgb 'blue' title 'Q(u)\n");// lt = linetype
	
			for(int l=layers-1; l >= 0; l--) {
				float d = far + float(l)*this->depthStep;// d is inverse depth
				float CostVolume_v = C_ptr[i+l*rows*cols];

				fprintf(gid, "%f %f\n", d, CostVolume_v);
				// fprintf(gid, "plot './uv.txt' with dots");
			}
			fprintf(gid, "e\n");// end
	// 	}
	// }
	cv::Mat Eaux_cpu;
	Eaux_.download(Eaux_cpu);
	float* Eaux_ptr = (float*)Eaux_cpu.data;
			for(int l=layers-1; l >= 0; l--) {
				float d = far + float(l)*this->depthStep;// d is inverse depth
				float Eaux_v = Eaux_ptr[i+l*rows*cols];

				fprintf(gid, "%f %f\n", d, Eaux_v);
				// fprintf(gid, "plot './uv.txt' with dots");
			}
			fprintf(gid, "e\n");// end
			

			for(int l=layers-1; l >= 0; l--) {
				float d = far + float(l)*this->depthStep;// d is inverse depth
				float CostVolume_v = C_ptr[i+l*rows*cols];
				float Eaux_v = Eaux_ptr[i+l*rows*cols];
				float Q_v = Eaux_v - CostVolume_v;

				fprintf(gid, "%f %f\n", d, Q_v);
				// fprintf(gid, "plot './uv.txt' with dots");
			}
			fprintf(gid, "e\n");// end

	pclose(gid);

}