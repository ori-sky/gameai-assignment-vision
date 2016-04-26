#include <iostream>
#include <fstream>

#ifdef __GNUC__
#include <experimental/filesystem> // Full support in C++17
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::tr2::sys;
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>

// https://msdn.microsoft.com/en-us/library/dn986850.aspx
// GCC 5.2.1 Ok on Linux
// g++ -std=c++11 facerec.cpp -lstdc++fs

int main(int argc, char *argv[]) {
	std::vector<cv::Mat> images;
	std::vector<int>     labels;

	// Iterate through all subdirectories, looking for .pgm files
	fs::path p(argc > 1 ? argv[1] : "../assets/faces");
	for(const auto &entry : fs::recursive_directory_iterator{p}) {
		if(fs::is_regular_file(entry.status())) {
			if(entry.path().extension() == ".pgm") {
				std::string str = entry.path().parent_path().stem().string();
				int label = atoi(str.c_str() + 1); // s1 -> 1
				images.push_back(cv::imread(entry.path().string().c_str(), 0));
				labels.push_back(label);
			}
		}
	}

	// Randomly choose an image, and remove it from the main collection
	std::srand(std::time(0));
	int rand_image_id = std::rand() % images.size();
	cv::Mat testSample = images[rand_image_id];
	int     testLabel  = labels[rand_image_id];
	images.erase(images.begin() + rand_image_id);
	labels.erase(labels.begin() + rand_image_id);
	std::cout << "actual class = " << testLabel << std::endl;
	std::cout << "training..." << std::endl;

	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	int predictedLabel = model->predict(testSample);
	std::cout << "predicted class = " << predictedLabel << std::endl;
	return 0;
}
