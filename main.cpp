#include <iostream>
#include <opencv2/opencv.hpp>
#include "json/single_include/nlohmann/json.hpp"
#include <chrono>// measure time // https://qiita.com/yukiB/items/01f8e276d906bf443356

#include "f_load.hpp"
#include "CostVolume.hpp"
#include "Regulariser.hpp"
#include "pcl3d.hpp"
 
int main (int argc, char* argv[])
{
	if (argc < 2){
		std::cout << "Usage Example: ./build/main input/json/fountain.json\n";
		std::cerr << "argc: " << argc << "should be 2\n";
		return 1;
	}	
	const std::string json_name = argv[1];
	std::ifstream ifs(json_name, std::ios::in);
	if(!ifs.is_open()){
		std::cerr << "Error, cannot open file, check argv: " << json_name << std::endl;
		return 1; 
	}
	nlohmann::json j_;
	ifs >> j_;
	
	// costvolume
	const float
		depth_min = j_["costvolume"]["depth_min"],
		depth_max = j_["costvolume"]["depth_max"];
	const float 
		near_ = 1.0f/depth_min,
		far_ = 1.0f/depth_max;
	const int 
		layers_ = j_["costvolume"]["layers_"], 
		frames_per_reference_image = j_["costvolume"]["frames_per_reference_image"];	
	const std::string similarity = j_["costvolume"]["similarity"];
	
	// regulariser
	const float 
		alphaG = j_["regulariser"]["alphaG"], 
		betaG = j_["regulariser"]["betaG"],
		theta_start = j_["regulariser"]["theta_start"], 
		theta_min = j_["regulariser"]["theta_min"],
		epsilon = j_["regulariser"]["epsilon"],
		lambda = j_["regulariser"]["lambda"],
		scale_Eaux = j_["regulariser"]["scale_Eaux"];
		
	// flag
	const bool 
		isRegulariser = j_["flag"]["isRegulariser"], 
		isResetCostVolume = j_["flag"]["isResetCostVolume"],
		isViewNoisyPoints = j_["flag"]["isViewNoisyPoints"];
	
	const float 
		width_init = j_["img_size"]["width_init"],
		height_init = j_["img_size"]["height_init"];
	const int 
		width = j_["img_size"]["width"], 
		height = j_["img_size"]["height"];
	const float
		fx_init = j_["K"]["fx_init"],
		fy_init = j_["K"]["fy_init"],
		cx_init = j_["K"]["cx_init"],
		cy_init = j_["K"]["cy_init"];
	const float 
		fx = fx_init * width/width_init,
		fy = fy_init * height/height_init,
		cx = cx_init * width/width_init,
		cy = cy_init * height/height_init;

	std::vector<cv::Mat> vec_R, vec_t;
	std::vector<std::string> vec_imgname;
	const std::string 
		filename = j_["file"]["filename"],
		dataset_name = j_["dataset_name"];

	if(dataset_name == "fountain") file_loader_fou(filename, vec_imgname, vec_R, vec_t);
	else if(dataset_name == "icl_nuim") file_loader_icl(filename, vec_imgname, vec_R, vec_t);
	else std::cerr << "not found dataset_name: " << dataset_name << std::endl;
	
	std::string dir_img = j_["file"]["dir_img"];

	const cv::Mat K_cpu = (cv::Mat_<float>(3,3) << fx, 0.0, cx,
											0.0, fy, cy,
											0.0, 0.0, 1.0);
	std::cout << "K: " << K_cpu << std::endl;
	
	const int rows_ = height, cols_ = width;// 480, 640;
	CostVolume costvolume(rows_, cols_, layers_, near_, far_, K_cpu, similarity);
	MyPCLViewer mypclviewer(rows_, cols_, layers_, near_, far_, K_cpu, dataset_name);
	Regulariser regulariser(rows_, cols_, layers_, near_, far_, alphaG, betaG, theta_start, theta_min, epsilon, lambda, scale_Eaux);
	
	cv::Mat src, Rwc, twc;
	for (std::size_t i = 0; i < vec_imgname.size(); i++){
		try {
			const std::string imgname = dir_img + vec_imgname[i];
			src = cv::imread(imgname, cv::IMREAD_COLOR);
			if (src.empty()) {
				std::cerr << "failed to load image. check path: " << imgname << std::endl;
				return 1;
			}
			CV_Assert(src.type() == CV_8UC3);// std::cout << src.type();// element type 16 = CV_8UC3// https://koshinran.hateblo.jp/entry/2017/10/30/200250
			cv::resize(src, src, cv::Size(width, height));            
			src.convertTo(src, CV_32FC3);// CV_8UC3 -> CV_32FC3
			cvtColor(src, src, CV_BGR2BGRA);// CV_32FC3 -> CV_32FC4 
		}
		catch(const cv::Exception& ex)
		{
			std::cout << "cv::Exception Error: " << ex.what() << std::endl;
		}
		Rwc = vec_R[i];
		twc = vec_t[i];
			
		if (costvolume.count_ == 0) {
			costvolume.set_refRGB(src, Rwc, twc);
			costvolume.imwrite_RGB(src);
			mypclviewer.set_refRGB(src);
			if(isRegulariser){
				regulariser.compute_gWeight(costvolume.reference_image_gray_);// cv::cuda::reference_image_gray_
				regulariser.imwrite_gWeight();
			}
		}
		else costvolume.update_CostVolume(src, Rwc, twc);

		costvolume.count_++;
		std::cout << "count: " << costvolume.count_ << std::endl;
		
		if (costvolume.count_ >= 2) {// because 1st iteration is for setting ref image
			costvolume.imwrite_RGB(src);
			costvolume.imwrite_inv_depth();
			// costvolume.debug_projection();
			// costvolume.debug_Cdata();
			// costvolume.plot_Cdata();
		}

		if (costvolume.count_ == frames_per_reference_image) {
			if(isRegulariser){
				std::chrono::system_clock::time_point start1, end1;
				start1 = std::chrono::system_clock::now(); 
				regulariser.optimize_global(costvolume.count_,
					costvolume.Cdata,
					costvolume.CminIdx,
					costvolume.Cmin,
					costvolume.Cmax);
				end1 = std::chrono::system_clock::now();
				double elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count(); 
				std::cout << "elapsed time(milliseconds): " << elapsed1 << std::endl;
				// regulariser.debug_cumat();
				regulariser.imwrite_depth_refined(costvolume.count_);
				if(regulariser.regularised_flag){
					std::cout << "point clouds WITH regularisation" << std::endl;
					cv::Mat inv_depth;
					regulariser.d_.download(inv_depth);
					// mypclviewer.visuPointsFromDepthMap_camera(inv_depth);
					mypclviewer.visuPointsFromDepthMap_world(inv_depth, vec_R, vec_t);
					// mypclviewer.visuSurfaceFromDepthMap_world(inv_depth, vec_R, vec_t);
				}
				else{
					std::cerr << "regularisation was not done" << std::endl;
				}
			}
			
			if(isViewNoisyPoints){
				std::cout << "point clouds WITHOUT regularisation --> noisy" << std::endl;
				cv::Mat inv_depth;
				costvolume.CminIdx.download(inv_depth);
				// mypclviewer.visuPointsFromDepthMap_camera(inv_depth);
				mypclviewer.visuPointsFromDepthMap_world(inv_depth, vec_R, vec_t);
				// mypclviewer.visuPointsFromGTDepthMap_world("input/fountain/gt/Depth0005.exr", vec_R, vec_t);
			}

			if(isResetCostVolume) costvolume.reset();
		}

		if (static_cast<int>(i) == frames_per_reference_image) break;
	}
	return 0;
}

