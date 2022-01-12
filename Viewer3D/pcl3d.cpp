#include "pcl3d.hpp"

MyPCLViewer::MyPCLViewer(int _rows, int _cols, int _layers, float _near, float _far, const cv::Mat& K_cpu, std::string dataset_name):
	rows(_rows), cols(_cols), layers(_layers), near(_near), far(_far), K_cpu(K_cpu), dataset_name(dataset_name)
{
	K_cpu_inv = K_cpu.inv();
	// point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>());
	// point_cloud_ptr_ = new pcl::PointCloud<pcl::PointXYZRGB>;	
}

void MyPCLViewer::set_refRGB(const cv::Mat& reference_image)
{
	this->reference_image = reference_image;
	CV_Assert(this->reference_image.rows == this->rows && this->reference_image.cols == this->cols);
	CV_Assert(this->reference_image.type() == CV_32FC4);	
}

void MyPCLViewer::visuPointsFromDepthMap_camera(const cv::Mat& inv_depth)
{	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(inv_depth.type() == CV_32FC1);	
	CV_Assert(this->K_cpu_inv.type() == CV_32FC1);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
			pcl::PointXYZRGB point;
			point.z = 1.0/inv_depth.at<float>(v,u);
			point.x = (K_cpu_inv.at<float>(0,0)*u + K_cpu_inv.at<float>(0,2)) * point.z;
			point.y = (K_cpu_inv.at<float>(1,1)*v + K_cpu_inv.at<float>(1,2)) * point.z;
			point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
			point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
			point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
			point_cloud_ptr_->points.push_back(point);
		}
	}
	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis(point_cloud_ptr_);
}

void MyPCLViewer::visuPointsFromDepthMap_world(const cv::Mat& inv_depth,
	const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t)
{	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(inv_depth.type() == CV_32FC1);	
	CV_Assert(this->K_cpu_inv.type() == CV_32FC1);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
			pcl::PointXYZRGB point;
			float xc, yc, zc;
			// zc = inv_depth.at<float>(v,u)<=0 ? 1.0/far:1.0/inv_depth.at<float>(v,u);
			// zc = inv_depth.at<float>(v,u)<=0 ? 0.0:1.0/inv_depth.at<float>(v,u);
			zc = inv_depth.at<float>(v,u)<far ? 0.0:1.0/inv_depth.at<float>(v,u);
			// if(u>300 && u < 310 && v > 300 && v < 305) std::cout << "depth: " << zc << std::endl;
			xc = (K_cpu_inv.at<float>(0,0)*u + K_cpu_inv.at<float>(0,2)) * zc;
			yc = (K_cpu_inv.at<float>(1,1)*v + K_cpu_inv.at<float>(1,2)) * zc;
			const cv::Mat Xc = (cv::Mat_<float>(3,1) << xc, yc, zc);
			// std::cout << "Xc: " << Xc << std::endl;
			const cv::Mat Rwc = vec_R[0];
			const cv::Mat twc = vec_t[0];
			const cv::Mat Xw = Rwc*Xc+twc;
			point.x = Xw.at<float>(0);
			point.y = Xw.at<float>(1);
			point.z = Xw.at<float>(2);
			point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
			point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
			point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
			point_cloud_ptr_->points.push_back(point);				
		}
	}
	// pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	// this->estimate_normals(point_cloud_ptr_, normals);
	
	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis_world(point_cloud_ptr_, vec_R, vec_t);
	// this->rgbVis_world_normals(point_cloud_ptr_, vec_R, vec_t, normals);
}

void MyPCLViewer::visuPointsFromGTDepthMap_world(std::string filename,
	const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t)
{	

	//// for Depth0001.exr
	cv::Mat img_depth=cv::imread(filename, cv::IMREAD_ANYDEPTH);
	if (img_depth.empty()) std::cerr << "failed to load image." << std::endl;
	cv::Mat depth_map;

	// extract B from BGR．
	std::vector<cv::Mat> planes;
	cv::split(img_depth, planes);
	depth_map = planes[0].clone();

	cv::resize(depth_map, depth_map ,cv::Size(cols, rows)); 

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(this->K_cpu_inv.type() == CV_32FC1);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
			pcl::PointXYZRGB point;
			float xc, yc, zc;
			const float val = depth_map.at<float>(v,u);
			zc = val>1./far ? 1./far:val;
			// if(u>300 && u < 310 && v > 300 && v < 305) std::cout << "depth: " << zc << std::endl;
			xc = (K_cpu_inv.at<float>(0,0)*u + K_cpu_inv.at<float>(0,2)) * zc;
			yc = (K_cpu_inv.at<float>(1,1)*v + K_cpu_inv.at<float>(1,2)) * zc;
			const cv::Mat Xc = (cv::Mat_<float>(3,1) << xc, yc, zc);
			// std::cout << "Xc: " << Xc << std::endl;
			const cv::Mat Rwc = vec_R[0];
			const cv::Mat twc = vec_t[0];
			const cv::Mat Xw = Rwc*Xc+twc;
			point.x = Xw.at<float>(0);
			point.y = Xw.at<float>(1);
			point.z = Xw.at<float>(2);
			point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
			point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
			point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
			point_cloud_ptr_->points.push_back(point);				
		}
	}
	// pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	// this->estimate_normals(point_cloud_ptr_, normals);
	

	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis_world(point_cloud_ptr_, vec_R, vec_t);
	// this->rgbVis_world_normals(point_cloud_ptr_, vec_R, vec_t, normals);
}

void MyPCLViewer::visuPointsFromXyz_camera(const cv::Mat& xyzImg_cpu)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(xyzImg_cpu.type() == CV_32FC4);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
				pcl::PointXYZRGB point;
				point.z = xyzImg_cpu.at<cv::Vec4f>(v,u)[2];
				point.x = xyzImg_cpu.at<cv::Vec4f>(v,u)[0];
				point.y = xyzImg_cpu.at<cv::Vec4f>(v,u)[1];
				if(xyzImg_cpu.at<cv::Vec4f>(v,u)[3] == 10){// outlier
					point.b = 255;
					point.g = 0;
					point.r = 0;
				}
				else{
					point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
					point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
					point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
				}
				point_cloud_ptr_->points.push_back(point);
		}
	}	
	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis(point_cloud_ptr_);
}


void MyPCLViewer::visuPointsFromXyz_world(const cv::Mat& xyzImg_cpu,
		const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(xyzImg_cpu.type() == CV_32FC4);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
				pcl::PointXYZRGB point;
				float xc, yc, zc;
				xc = xyzImg_cpu.at<cv::Vec4f>(v,u)[0];
				yc = xyzImg_cpu.at<cv::Vec4f>(v,u)[1];
				zc = xyzImg_cpu.at<cv::Vec4f>(v,u)[2];
				const cv::Mat Xc = (cv::Mat_<float>(3,1) << xc, yc, zc);
				// std::cout << "Xc: " << Xc << std::endl;
				const cv::Mat Rwc = vec_R[0];
				const cv::Mat twc = vec_t[0];
				const cv::Mat Xw = Rwc*Xc+twc;

				point.x = Xw.at<float>(0);//[0];
				point.y = Xw.at<float>(1);//[1];
				point.z = Xw.at<float>(2);//[2];
				// std::cout << "Xw: " << Xw << std::endl;
				// std::cout << "point.x: " << point.x << std::endl;
				// std::cout << "point.y: " << point.y << std::endl;
				// std::cout << "point.z: " << point.z << std::endl;
				if(xyzImg_cpu.at<cv::Vec4f>(v,u)[3] == 10){// outlier
					point.b = 255;
					point.g = 0;
					point.r = 0;
				}
				else{
					point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
					point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
					point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
				}
				point_cloud_ptr_->points.push_back(point);
		}
	}
	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis_world(point_cloud_ptr_, vec_R, vec_t);
}

void MyPCLViewer::visuSurfaceFromDepthMap_world(const cv::Mat& inv_depth,
	const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t)
{	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);
	point_cloud_ptr_->points.clear();
	point_cloud_ptr_->width = cols;
	point_cloud_ptr_->height = rows;
	CV_Assert(inv_depth.type() == CV_32FC1);	
	CV_Assert(this->K_cpu_inv.type() == CV_32FC1);	
	CV_Assert(this->reference_image.type() == CV_32FC4);	
	for(int v=0; v<rows; v++) {
		for(int u=0; u<cols; u++) {
			pcl::PointXYZRGB point;
			float xc, yc, zc;
			zc = inv_depth.at<float>(v,u)<far ? 0.0:1.0/inv_depth.at<float>(v,u);
			xc = (K_cpu_inv.at<float>(0,0)*u + K_cpu_inv.at<float>(0,2)) * zc;
			yc = (K_cpu_inv.at<float>(1,1)*v + K_cpu_inv.at<float>(1,2)) * zc;
			const cv::Mat Xc = (cv::Mat_<float>(3,1) << xc, yc, zc);
			const cv::Mat Rwc = vec_R[0];
			const cv::Mat twc = vec_t[0];
			const cv::Mat Xw = Rwc*Xc+twc;
			point.x = Xw.at<float>(0);
			point.y = Xw.at<float>(1);
			point.z = Xw.at<float>(2);
			point.b = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[0]);
			point.g = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[1]);
			point.r = static_cast<std::uint8_t>(reference_image.at<cv::Vec4f>(v,u)[2]);
			point_cloud_ptr_->points.push_back(point);				
		}
	}
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	this->estimate_normals(point_cloud_ptr_, normals);
	this->savePoints(point_cloud_ptr_, "pcl.ply");
	this->rgbVis_world_surface(point_cloud_ptr_, vec_R, vec_t, normals);
}


void MyPCLViewer::rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_){
	// https://pcl.readthedocs.io/projects/tutorials/en/pcl-1.11.0/pcl_visualizer.html
	// creates the viewer object
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));
	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));
	
	// point_cloud_ptr_->points.push_back(pcl::PointXYZRGB());
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_);

	viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr_, rgb, "projected_depth_image");

	// add pcl in viewer
	// viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "rgb cloud");
	// viewer->addPointCloud<pcl::PointXYZRGB> (point_cloud_ptr_, rgb, "rgb cloud");
	// background RGB colour
	viewer->setBackgroundColor(0, 0, 0);
	// the size of the rendered points is 3.0
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_depth_image");
	// The size of the cylinders(each XYZ axis) is 1.0(default)
	viewer->addCoordinateSystem(1.0);
	// viewer->addCoordinateSystem(1.0, 3, 3, 3, "cam1");
	// sets up some handy camera parameters to make things look nice.
	viewer->initCameraParameters();
	// viewer->setCameraPosition(0, 0, 0,
	viewer->setCameraPosition(-0.0067756, -0.0685564, -0.462478,
												 0,          0,         1,
								-0.0105255, -0.9988450,  0.0468715);
									// 0, -1, 0);

	// viewer->setCameraClipDistances(0.0186334, 18.6334);
	viewer->setCameraClipDistances(1/near, 1/far);
	viewer->setCameraFieldOfView(0.8575);

	// bool updating_pointcloud_ = true;
	// while(!viewer->wasStopped()) {
	// 	viewer->spinOnce (100);

	// 	boost::mutex::scoped_lock update_lock(update_pc_mutex_);
	// 	// if(updating_pointcloud_) {
	// 	// 	if(!viewer->updatePointCloud(point_cloud_ptr_, "projected_depth_image"))
	// 	// 		// viewer->addPointCloud(point_cloud_ptr_, rgb, "projected_depth_image");
	// 	// 		viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr_, rgb, "projected_depth_image");
	// 	// 	updating_pointcloud_ = false;

	// 	// }
	// 	update_lock.unlock();
	// }
	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		// std::this_thread::sleep_for(100ms);
	}
}

	/*
	// // https://gist.github.com/YHaruoka/6a8bd64dbc25beb6d161ff99d80239c3
	// // http://tecsingularity.com/pcl/make_point_cloud/
	*/


void MyPCLViewer::rgbVis_world(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_,
			const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t){
	// https://pcl.readthedocs.io/projects/tutorials/en/pcl-1.11.0/pcl_visualizer.html
	// creates the viewer object
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_);
	viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr_, rgb, "projected_depth_image");

	// background RGB colour, (0,0,0) is black
	viewer->setBackgroundColor(0, 0, 0);
	// the size of the rendered points is 3.0
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_depth_image");
	// The size of the cylinders(each XYZ axis) is 2.0(default: 1.0)
	viewer->addCoordinateSystem(2.0, 0, 0, 0, "world origin");// (x,y,z)=(0,0,0)
	viewer->initCameraParameters();
	const std::uint8_t max_iter = 35;
	for (std::uint8_t i = 0; i < vec_t.size(); i++){
		const cv::Mat twc = vec_t[i];
		// float tx, ty, tz;
		// tx = twc.at<float>(0, 0);//(v,u)
		// ty = twc.at<float>(1, 0); 
		// tz = twc.at<float>(2, 0);
		// viewer->addCoordinateSystem(1.0, tx, ty, tz, "cam"+std::to_string(i));

		const cv::Mat Rwc = vec_R[i];
		const cv::Mat Twc = (cv::Mat_<float>(4,4) <<
			 Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2), twc.at<float>(0, 0),
			 Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2), twc.at<float>(1, 0),
			 Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2), twc.at<float>(2, 0),//);
			 0,0,0,1);

		Eigen::Matrix4f eigenMat;
		cv::cv2eigen(Twc, eigenMat);
		// Eigen::Affine3f view;
		// view = eigenMat;
		Eigen::Affine3f viewer_pose(eigenMat);
		viewer->addCoordinateSystem(1.0, viewer_pose, "cam"+std::to_string(i));

		// https://www.codetd.com/ja/article/6553570
		// Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);// (0,0,0) is camera position in camera frame, Tcw transforms camera frame into world frame, pos_vector locates in world frame
		Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(-1, -3, -5);
		// Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;// blue(Z) axis is the depth direction
		Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0.1, 0.3, 1) + pos_vector;
		Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);// default green(Y) axis heads down(y=1), so camera head is up(y=-1)
		if(dataset_name == "icl_nuim") {
			pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
			look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
			up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 1, 0);
		}
		viewer->setCameraPosition(
			pos_vector[0], pos_vector[1], pos_vector[2],
			look_at_vector[0], look_at_vector[1], look_at_vector[2],
			up_vector[0], up_vector[1], up_vector[2]);
		
		// Eigen::Matrix3f intrinsics;
		// cv::cv2eigen(this->K_cpu, intrinsics);
		// viewer->setCameraParameters(intrinsics, eigenMat);

		if(i >= max_iter) break;
	}
	/*
	// sets up some handy camera parameters to make things look nice.
	viewer->initCameraParameters();
	const cv::Mat twc = vec_t[0];
	float tx, ty, tz;
	tx = twc.at<float>(0, 0);//(v,u)
	ty = twc.at<float>(1, 0);
	tz = twc.at<float>(2, 0);
	viewer->setCameraPosition(tx, ty, tz,
												 0, 0, 1,//0,          0,         1,
									0, -1, 0);
	*/


	
	// float px, py, pz;
	// px = point_cloud_ptr_->points[0].x;
	// py = point_cloud_ptr_->points[0].y;
	// pz = point_cloud_ptr_->points[0].z;
	// float vx, vy, vz;
	// vx = px;//-tx;// vector AP = P - A
	// vy = py;//-ty;
	// vz = pz;//-tz;
	

	// const cv::Mat v1 = (cv::Mat_<float>(3,1) <<
	// 			   0, 0, 1);
	// const cv::Mat Rwc = vec_R[0];
	// const cv::Mat vray = Rwc.inv()*v1;
	// float vx1, vy1, vz1;
	// vx1 = twc.at<float>(0, 0);//(v,u)
	// vy1 = twc.at<float>(1, 0);
	// vz1 = twc.at<float>(2, 0);


	
	


	// viewer->setCameraClipDistances(0.0186334, 18.6334);
	viewer->setCameraClipDistances(1/near, 1/far);
	viewer->setCameraFieldOfView(0.8575);

	while (!viewer->wasStopped ())// type "q" to stop
	{
		viewer->spinOnce (100);// 100ms loop
	}
}


void MyPCLViewer::rgbVis_world_normals(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_,
			const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t,
			pcl::PointCloud<pcl::Normal>::ConstPtr normals
			){
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_);
	viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr_, rgb, "projected_depth_image");
	viewer->setBackgroundColor(0, 0, 0);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_depth_image");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(point_cloud_ptr_, normals, 10, 0.05, "normals");
	viewer->addCoordinateSystem(2.0, 0, 0, 0, "world origin");// (x,y,z)=(0,0,0)
	viewer->initCameraParameters();
	for (std::uint8_t i = 0; i < vec_t.size(); i++){
		const cv::Mat twc = vec_t[i];
		const cv::Mat Rwc = vec_R[i];
		const cv::Mat Twc = (cv::Mat_<float>(4,4) <<
			 Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2), twc.at<float>(0, 0),
			 Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2), twc.at<float>(1, 0),
			 Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2), twc.at<float>(2, 0),//);
			 0,0,0,1);

		Eigen::Matrix4f eigenMat;
		cv::cv2eigen(Twc, eigenMat);
		Eigen::Affine3f viewer_pose(eigenMat);
		viewer->addCoordinateSystem(1.0, viewer_pose, "cam"+std::to_string(i));
		Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
		Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
		Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);// default green(Y) axis heads down(y=1), so camera head is up(y=-1)
		viewer->setCameraPosition(
			pos_vector[0], pos_vector[1], pos_vector[2],
			look_at_vector[0], look_at_vector[1], look_at_vector[2],
			up_vector[0], up_vector[1], up_vector[2]);	
			
	}
	
	viewer->setCameraClipDistances(1/near, 1/far);
	viewer->setCameraFieldOfView(0.8575);

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);// 100ms loop
	}
}


void MyPCLViewer::rgbVis_world_surface(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_,
			const std::vector<cv::Mat>& vec_R, const std::vector<cv::Mat>& vec_t,
			pcl::PointCloud<pcl::Normal>::ConstPtr normals){
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Projected Depth Image"));
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr_);
	viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud_ptr_, rgb, "projected_depth_image");
	viewer->setBackgroundColor(0, 0, 0);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "projected_depth_image");
	// viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(point_cloud_ptr_, normals, 10, 0.05, "normals");
	viewer->addCoordinateSystem(2.0, 0, 0, 0, "world origin");// (x,y,z)=(0,0,0)
	viewer->initCameraParameters();
	for (std::uint8_t i = 0; i < vec_t.size(); i++){
		const cv::Mat twc = vec_t[i];
		const cv::Mat Rwc = vec_R[i];
		const cv::Mat Twc = (cv::Mat_<float>(4,4) <<
			 Rwc.at<float>(0,0), Rwc.at<float>(0,1), Rwc.at<float>(0,2), twc.at<float>(0, 0),
			 Rwc.at<float>(1,0), Rwc.at<float>(1,1), Rwc.at<float>(1,2), twc.at<float>(1, 0),
			 Rwc.at<float>(2,0), Rwc.at<float>(2,1), Rwc.at<float>(2,2), twc.at<float>(2, 0),//);
			 0,0,0,1);

		Eigen::Matrix4f eigenMat;
		cv::cv2eigen(Twc, eigenMat);
		Eigen::Affine3f viewer_pose(eigenMat);
		viewer->addCoordinateSystem(1.0, viewer_pose, "cam"+std::to_string(i));
		Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
		Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
		Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);// default green(Y) axis heads down(y=1), so camera head is up(y=-1)
		viewer->setCameraPosition(
			pos_vector[0], pos_vector[1], pos_vector[2],
			look_at_vector[0], look_at_vector[1], look_at_vector[2],
			up_vector[0], up_vector[1], up_vector[2]);		
	}
	
	viewer->setCameraClipDistances(1/near, 1/far);
	viewer->setCameraFieldOfView(0.8575);

	// pcl::Poisson<pcl::PointNormal> poisson;
	pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
   poisson.setDepth(9);
	// pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
   pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
   // cloud_with_normals = cloud + normals
	pcl::concatenateFields (*point_cloud_ptr_, *normals, *cloud_with_normals);
   poisson.setInputCloud(cloud_with_normals);
   pcl::PolygonMesh mesh;
   poisson.reconstruct(mesh);

	viewer->addPolygonMesh(mesh, "polygon");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING,
                                     pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "polygon");

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);// 100ms loop
	}
}

void MyPCLViewer::savePoints(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_, std::string filename){
	std::cout << "total number of points: " << point_cloud_ptr_->size() << std::endl;
	std::cout << "Save PointsCloud as "<< filename << std::endl;
	
	// ply
	pcl::PLYWriter ply_writer;
	ply_writer.write<pcl::PointXYZRGB> (filename, *point_cloud_ptr_);

	// pcd
	// pcl::PCDWriter pcd_writer;
	// pcd_writer.write<pcl::PointXYZRGB> ("pcl.pcd", *point_cloud_ptr_);
}

void MyPCLViewer::estimate_normals(
	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_,
	pcl::PointCloud<pcl::Normal>::Ptr normals){
	// http://virtuemarket-lab.blogspot.com/2015/02/blog-post_35.html
	// https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation_using_integral_images.html


	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud (point_cloud_ptr_);//法線の計算を行いたい点群を指定する
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());//KDTREEを作る
	ne.setSearchMethod (tree);//検索方法にKDTREEを指定する
	ne.setRadiusSearch (0.5);//検索する半径を指定する, 0.5m
	ne.compute (*normals);//法線情報の出力先を指定する

	const int i = 5;
	std::cout << "debug; x: " << normals->points[i].normal_x << ", y: " << normals->points[i].normal_y << ", z: " << normals->points[i].normal_z << std::endl;


}


// void MyPCLViewer::reconstruct_surface(
// 	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr_,
// 	pcl::PointCloud<pcl::Normal>::Ptr normals,){
// 	pcl::Poisson<pcl::PointNormal> poisson;
//    poisson.setDepth(9);
// 	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal> ());
//    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
//    // cloud_with_normals = cloud + normals
//    poisson.setInputCloud(cloud_smoothed_normals);
//    PolygonMesh mesh;
//    poisson.reconstruct(mesh);
// 	// viewer->addPolygonMesh(mesh, "polygon");
// 	// viewer->setShapeRenderingProperties(PCL_VISUALIZER_SHADING,
//  //                                     PCL_VISUALIZER_SHADING_PHONG, "polygon");
// }
