// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include "caffe/caffe.hpp"

// for read and write image
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace caffe;  // NOLINT(build/namespaces)

enum OUTPUT_MODE {DRAW, LABEL, BOTH};

int main(int argc, char** argv) {
  if (argc < 5 || argc > 8) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations output_dir"
				<< " [DRAW/LABEL/BOTH]"
        << " [CPU/GPU] [Device ID]";
    return 1;
  }
  Caffe::set_phase(Caffe::TEST);
	OUTPUT_MODE mode = BOTH;
	if(argc >= 6){
		if(strcmp(argv[5], "DRAW") == 0){
			mode = DRAW;
		}else if(strcmp(argv[5], "LABEL") == 0){
			mode = LABEL;
		}
	}
  if (argc >= 7 && strcmp(argv[6], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 8) {
      device_id = atoi(argv[7]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  Net<float> caffe_test_net(argv[1]);
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  int total_iter = atoi(argv[3]);
	string output_dir(argv[4]);
	//creat output_dir
	string mkdir_cmd("mkdir " + output_dir);
	int found = mkdir_cmd.find("/");
	while(found > 0){// replace invalid "/" with "\" supported by system command
		mkdir_cmd.replace(found, 1, "\\");
		found = mkdir_cmd.find("/");
	}
	system(mkdir_cmd.c_str());

  LOG(ERROR) << "Running " << total_iter << " iterations.";
	vector<Blob<float>*> input_vec;
	int image_index = 0;
	const vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	ImageDataLayer<float>* image_layer = reinterpret_cast<ImageDataLayer<float>*>(layers[0].get());
	CHECK(image_layer);
	int new_height = image_layer->layer_param().image_data_param().new_height();
	int new_width = image_layer->layer_param().image_data_param().new_width();

  for (int i = 0; i < total_iter; ++i) {
		caffe_test_net.Forward(input_vec);
		const shared_ptr<Blob<float> > blob_ = caffe_test_net.blob_by_name("fc_8_det");
		const float* det_points = blob_->mutable_cpu_data();
		for(int n=0; n<blob_->num(); n++){
			det_points = blob_->mutable_cpu_data() + blob_->offset(n);
			string image_path(image_layer->lines_[image_index].first);
			found = image_path.find_last_of("/\\");
			string fn(image_path.substr(found+1));
			//LOG(ERROR) << output_dir << "/" << fn << " " \
				<< det_points[0] << " " << det_points[1] << " " << det_points[2] << " " << det_points[3] << "\n";
			int x1 = det_points[0];
			int y1 = det_points[1];
			int x2 = det_points[2];
			int y2 = det_points[3];
			cv::Mat cv_img;
			cv_img = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
			CHECK(!cv_img.empty());
			if(new_height > 0 && new_width > 0){
				// rescale points
				x1 = float(x1)/new_width * cv_img.cols;
				y1 = float(y1)/new_height * cv_img.rows;
				x2 = float(x2)/new_width * cv_img.cols;
				y2 = float(y2)/new_height * cv_img.rows;
			}
			// draw boundary in image
			if(mode==BOTH || mode==DRAW){
				// write image with boundary
				line(cv_img, cv::Point(x1, y1), cv::Point(x2, y1), cv::Scalar(0, 0, 255), 3);
				line(cv_img, cv::Point(x2, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
				line(cv_img, cv::Point(x2, y2), cv::Point(x1, y2), cv::Scalar(0, 0, 255), 3);
				line(cv_img, cv::Point(x1, y2), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 3);
				// write image
				string out_path(output_dir+"/"+fn);
				CHECK_EQ(imwrite(out_path, cv_img), true) << "write image " + out_path + " failed";
			}
			if(mode==BOTH || mode==LABEL){
				// write image predict label
				std::fstream fs;
				found = fn.find(".");
				string fpath(output_dir+"/"+fn.substr(0,found)+".lbl");
				fs.open(fpath, std::ios::out);
				if(fs.is_open()){
					fs << x1 << " " << y1 << " " << x2 << " " << y2; 
					fs.close();
				}else{
					LOG(ERROR) << "Can't create " << fpath << "\n";
				}
			}

			LOG(ERROR) << "[" << image_index+1 << "/" << image_layer->lines_.size() << "]" << " " << fn << "\n";
			image_index ++;
			if(image_index >= image_layer->lines_.size()){
				LOG(ERROR) << "restart from begining\n";
				image_index = 0;
			}
		}
  }

  return 0;
}
