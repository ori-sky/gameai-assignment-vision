#include <iostream>
#include <fstream>

#ifdef __GNUC__
#include <experimental/filesystem> // Full support in C++17
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::tr2::sys;
#endif

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>

#define CONFIG_NUM_WORKERS 2
#define CONFIG_FACE_CLASS 41

// https://msdn.microsoft.com/en-us/library/dn986850.aspx
// GCC 5.2.1 Ok on Linux
// g++ -std=c++11 facerec.cpp -lstdc++fs

cv::Ptr<cv::face::BasicFaceRecognizer> make_recognizer(int argc, char *argv[]) {
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

	std::cout << "training..." << std::endl;
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	return model;
}

void camera_loop(boost::shared_ptr<boost::asio::io_service> io_service,
                 cv::VideoCapture vid,
                 cv::Ptr<cv::face::BasicFaceRecognizer> recognizer) {
	cv::Mat frame;
	vid >> frame;

	int new_width, new_height;
	float actual_aspect = (float)frame.cols / (float)frame.rows;
	float target_aspect = 92.0f / 112.0f;
	if(actual_aspect > target_aspect) {
		new_height = frame.rows;
		new_width = new_height * target_aspect;
	} else {
		new_width = frame.cols;
		new_height = new_width / target_aspect;
	}

	cv::Rect rect(frame.cols / 2 - new_width / 2, frame.rows / 2 - new_height / 2,
	              new_width, new_height);

	cv::Mat gray_frame, resized_frame;
	cv::cvtColor(frame(rect), gray_frame, CV_RGB2GRAY);
	cv::resize(gray_frame, resized_frame, cv::Size(92, 112));

	int predicted = recognizer->predict(resized_frame);
	if(predicted == CONFIG_FACE_CLASS) {
		std::cout << "recognized face" << std::endl;
	} else {
		std::cout << std::endl;
	}

	cv::imshow("optflow", resized_frame);
	cv::waitKey(1000/30);
	//if(cv::waitKey(30) < 0) {
		io_service->post(std::bind(camera_loop, io_service, vid, recognizer));
	//}
}

void camera_main(boost::shared_ptr<boost::asio::io_service> io_service,
                 cv::Ptr<cv::face::BasicFaceRecognizer> recognizer) {
	cv::VideoCapture vid(0);
	if(!vid.isOpened()) {
		throw std::runtime_error("failed to open video capture device");
	}
	cv::namedWindow("optflow");
	io_service->post(std::bind(camera_loop, io_service, vid, recognizer));
}

void worker_main(boost::shared_ptr<boost::asio::io_service> io_service) {
	io_service->run();
}

int main(int argc, char *argv[]) {
	auto recognizer = make_recognizer(argc, argv);

	auto io_service = boost::make_shared<boost::asio::io_service>();
	auto work       = boost::make_shared<boost::asio::io_service::work>(*io_service);
	auto strand     = boost::make_shared<boost::asio::io_service::strand>(*io_service);
	boost::thread_group workers;
	for(unsigned int w = 0; w < CONFIG_NUM_WORKERS; ++w) {
		workers.create_thread(boost::bind(worker_main, io_service));
	}
	io_service->post(boost::bind(camera_main, io_service, recognizer));
	work.reset();
	workers.join_all();
	return 0;
}
