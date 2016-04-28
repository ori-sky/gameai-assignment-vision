#include <iostream>
#include <fstream>

#ifdef __GNUC__
#include <experimental/filesystem>
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

cv::Ptr<cv::face::BasicFaceRecognizer> make_recognizer(int argc, char *argv[]) {
	std::vector<cv::Mat> images;
	std::vector<int>     labels;

	// iterate through subdirectories to locate .pgm files
	fs::path p(argc > 1 ? argv[1] : "../assets/faces");
	for(const auto &entry : fs::recursive_directory_iterator{p}) {
		if(fs::is_regular_file(entry.status())) {
			if(entry.path().extension() == ".pgm") {
				auto str = entry.path().parent_path().stem().string();
				int label = atoi(str.c_str() + 1);
				images.push_back(cv::imread(entry.path().string().c_str(), 0));
				labels.push_back(label);
			}
		}
	}

	auto recognizer = cv::face::createEigenFaceRecognizer();
	recognizer->train(images, labels);
	return recognizer;
}

void camera_loop(boost::shared_ptr<boost::asio::io_service> service,
                 cv::VideoCapture vid,
                 cv::Ptr<cv::face::BasicFaceRecognizer> recognizer,
                 unsigned int count) {
	cv::Mat frame;
	vid >> frame;

	/* calculate maximum dimensions matching target aspect ratio of 92x112 */
	float actual_aspect = (float)frame.cols / (float)frame.rows;
	float target_aspect = 92.0f / 112.0f;
	int target_width = frame.cols, target_height = frame.rows;
	if(actual_aspect > target_aspect) {
		target_width = target_height * target_aspect;
	} else {
		target_height = target_width / target_aspect;
	}

	/* crop frame using frame center as origin */
	cv::Rect rect(frame.cols / 2 - target_width / 2,
	              frame.rows / 2 - target_height / 2,
	              target_width, target_height);
	cv::Mat cropped_frame = frame(rect);

	/* convert to grayscale and scale to 92x112 */
	cv::Mat gray_frame, resized_frame;
	cv::cvtColor(cropped_frame, gray_frame, CV_RGB2GRAY);
	cv::resize(gray_frame, resized_frame, cv::Size(92, 112));

	/* keep running counter of successful recognitions
	 * decrement counter per unsuccessful frame to allow for partial error
	 */
	int predicted = recognizer->predict(resized_frame);
	if(predicted == CONFIG_FACE_CLASS) {
		if(count < 10) { ++count; }
	} else {
		if(count > 0) { --count; }
	}

	/* red    = not recognized
	 * yellow = partially recognized
	 * green  = fully recognized
	 */
	cv::Scalar color(0, 255, 255);
	switch(count) {
	case 0:
		color = cv::Scalar(0, 0, 255);
		break;
	case 10:
		color = cv::Scalar(0, 255, 0);
		break;
	}

	/* flip frame to display in a more sensible manner */
	cv::Mat flipped_frame;
	cv::flip(cropped_frame, flipped_frame, 1);

	/* draw colored border around frame to be displayed
	 * border surrounded by 2px of black to strengthen border contrast
	 */
	cv::Rect color_rect(0, 0, flipped_frame.cols, flipped_frame.rows);
	cv::rectangle(flipped_frame, color_rect, cv::Scalar(0), 10);
	cv::rectangle(flipped_frame, color_rect, color, 8);
	cv::rectangle(flipped_frame, color_rect, cv::Scalar(0), 2);

	cv::imshow("optflow", flipped_frame);
	cv::waitKey(1000/60);
	service->post(std::bind(camera_loop, service, vid, recognizer, count));
}

void camera_main(boost::shared_ptr<boost::asio::io_service> service,
                 cv::Ptr<cv::face::BasicFaceRecognizer> recognizer) {
	cv::VideoCapture vid(0);
	if(!vid.isOpened()) {
		throw std::runtime_error("failed to open video capture device");
	}
	cv::namedWindow("optflow");
	service->post(std::bind(camera_loop, service, vid, recognizer, 0));
}

void worker_main(boost::shared_ptr<boost::asio::io_service> service) {
	service->run();
}

int main(int argc, char *argv[]) {
	std::cout << "training..." << std::endl;
	auto recognizer = make_recognizer(argc, argv);
	std::cout << "done" << std::endl;

	auto service = boost::make_shared<boost::asio::io_service>();
	auto work    = boost::make_shared<boost::asio::io_service::work>(*service);
	auto strand  = boost::make_shared<boost::asio::io_service::strand>(*service);
	boost::thread_group workers;
	for(unsigned int w = 0; w < CONFIG_NUM_WORKERS; ++w) {
		workers.create_thread(boost::bind(worker_main, service));
	}
	service->post(boost::bind(camera_main, service, recognizer));
	work.reset();
	workers.join_all();
	return 0;
}
