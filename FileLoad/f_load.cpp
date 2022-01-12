#include "f_load.hpp"

void file_loader_fou (const std::string &filename, 
	std::vector<std::string> &vec_imgname,  
	std::vector<cv::Mat> &vec_R, 
	std::vector<cv::Mat> &vec_t)
{
	// https://qiita.com/Reed_X1319RAY/items/098596cda78e9c1a6bad
	std::ifstream ifs(filename, std::ios::in);
	if(!ifs.is_open()){
		std::cerr << "Error, cannot open file, check argv: " << filename << std::endl;
		std::exit(1); 
	}
   std::string line;
   // skip 2 line
   for(int i = 0; i < 2; i++){
   	std::getline(ifs, line);
   }
   while (std::getline(ifs, line)){
   	std::stringstream ss(line);// ss << line;
   	std::string imgname;
	   float R00, R01, R02, R10, R11, R12, R20, R21, R22, tx, ty, tz;
	   ss >> imgname >> R00 >> R01 >> R02 >> R10 >> R11 >> R12 >> R20 >> R21 >> R22 >> tx >> ty >> tz;
      cv::Mat Rwc = (cv::Mat_<float>(3,3) <<
				   R00, R01, R02,
				   R10, R11, R12,
				   R20, R21, R22);
		cv::Mat twc = (cv::Mat_<float>(3,1) <<
				   tx, ty, tz);
		vec_R.push_back(Rwc);
		vec_t.push_back(twc);
		vec_imgname.push_back(imgname);
	}
	ifs.close();
}

void file_loader_icl (const std::string &filename,
	std::vector<std::string> &vec_imgname,  
	std::vector<cv::Mat> &vec_R, 
	std::vector<cv::Mat> &vec_t)
{
	// https://qiita.com/Reed_X1319RAY/items/098596cda78e9c1a6bad
	std::ifstream ifs(filename, std::ios::in);
	if(!ifs.is_open()){
		std::cerr << "Error, cannot open file, check argv: " << filename << std::endl;
		std::exit(1); 
	}
   std::string line;
   while (std::getline(ifs, line)){
   	std::stringstream ss(line);// ss << line;
   	std::string tf_stamp, imgname;
	   float R00, R01, R02, R10, R11, R12, R20, R21, R22;//, tx, ty, tz;
	   float x,y,z,w, tx, ty, tz;
	   ss >> tf_stamp >> tx >> ty >> tz >> x >> y >> z >> w;
	   imgname = "rgb/" + tf_stamp + ".png";
	   
      float Nq = w * w + x * x + y * y + z * z;
      float s = 2./Nq;
      // std::cout << "s:" << s << std::endl;

      R00 = 1. - s * ( y*y + z*z );
      R01 = s * ( x*y - w*z);
      R02 = s * ( x*z + w*y);
      R10 = s * ( x*y + w*z );
      R11 = 1. - s * ( x*x + z*z );
      R12 = s * ( y*z - w*x );
      R20 = s * ( x*z - w*y );
      R21 = s * ( y*z + w*x );
      R22 = 1. - s * ( x*x + y*y );
      

      cv::Mat Rcw = (cv::Mat_<float>(3,3) <<
				   R00, R01, R02,
				   R10, R11, R12,
				   R20, R21, R22);
		cv::Mat tcw = (cv::Mat_<float>(3,1) <<
				   tx, ty, tz);
		// std::cout << "debug Rcw: " << Rcw << std::endl;
		// std::cout << "debug tcw: " << tcw << std::endl;
		vec_R.push_back(Rcw);// Rwc, not Rcw
		vec_t.push_back(tcw);// twc
		// cv::Mat Rwc =  Rcw.t();
		// cv::Mat twc = -Rcw.t()*tcw;
		// std::cout << "debug Rwc: " << Rwc << std::endl;
		// std::cout << "debug twc: " << twc << std::endl;
		// vec_R.push_back(Rwc);
		// vec_t.push_back(twc);
		vec_imgname.push_back(imgname);
	}
	ifs.close();
}


// int main(){
// 	const char *filename = "rgb.txt";
// 	std::vector<std::string> vec;
// 	file_loader (filename, vec);
// 	std::cout << vec[0] << std::endl;
// 	std::cout << vec.size() << std::endl;
//    return 0;
// }