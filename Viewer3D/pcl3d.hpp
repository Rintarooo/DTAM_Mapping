#ifndef PCL_H
#define PCL_H 

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>

#include <pcl/common/common_headers.h>
// #include <pcl/visualization/cloud_viewer.h>// show only pcl
#include <pcl/visualization/pcl_visualizer.h>// show pcl+normal+mesh
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/features/normal_3d.h>// for normal estimation

// convert between eigen and mat
#include <opencv2/core/eigen.hpp>

// surface
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/common/io.h> // for concatenateFields

class MyPCLViewer
{
public:
	// MyPCLViewer() {};
	MyPCLViewer(int _rows, int _cols, int _layers, float _near, float _far, const cv::Mat& Kcpu, std::string dataset_name);
	~MyPCLViewer() {};
  
	void set_refRGB(const cv::Mat&);
	void visuPointsFromDepthMap_camera(const cv::Mat&);
	void visuPointsFromDepthMap_world(const cv::Mat&,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&);
	void visuPointsFromGTDepthMap_world(std::string,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&);


	void visuPointsFromXyz_camera(const cv::Mat&);
	void visuPointsFromXyz_world(const cv::Mat&,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&);

	void visuSurfaceFromDepthMap_world(const cv::Mat&,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&);

private:
	const int rows, cols, layers;
	const float near, far;
	const cv::Mat K_cpu;
	const std::string dataset_name;
	cv::Mat K_cpu_inv, reference_image;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_;
	// boost::mutex update_pc_mutex_;

	void rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr);
	void rgbVis_world(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&);

	void rgbVis_world_normals(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&,
		pcl::PointCloud<pcl::Normal>::ConstPtr);
	void rgbVis_world_surface(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr,
		const std::vector<cv::Mat>&, const std::vector<cv::Mat>&,
		pcl::PointCloud<pcl::Normal>::ConstPtr);
	void estimate_normals(
		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr,
		pcl::PointCloud<pcl::Normal>::Ptr);

	void savePoints(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr, std::string filename);

};

#endif