#ifndef FILELOADER_H
#define FILELOADER_H 
#include <iostream>
#include <string>    // string
#include <fstream>   // ifstream, ofstream
#include <sstream>   // stringstream
#include <vector>
#include <opencv2/opencv.hpp>
#include <iomanip> // https://stackoverflow.com/questions/225362/convert-a-number-to-a-string-with-specified-length-in-c


void file_loader_fou (const std::string&,
	std::vector<std::string>&,  
	std::vector<cv::Mat>&, 
	std::vector<cv::Mat>&);

void file_loader_tum (const std::string&,
	std::vector<std::string>&,  
	std::vector<cv::Mat>&, 
	std::vector<cv::Mat>&);

void file_loader_icl (const std::string&,
	std::vector<std::string>&,  
	std::vector<cv::Mat>&, 
	std::vector<cv::Mat>&);

void file_loader_aha (const std::string&,
	std::vector<std::string>&,  
	std::vector<cv::Mat>&, 
	std::vector<cv::Mat>&);

#endif