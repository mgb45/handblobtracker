#ifndef __HANDTRACKER
#define __HANDTRACKER

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/RegionOfInterest.h"
#include "geometry_msgs/Point.h"
#include "facetracking/ROIArray.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sstream>
#include <string>
#include <ros/package.h>
#include "handblobtracker/HFPose2D.h"
#include "handblobtracker/HFPose2DArray.h"
#include <opencv2/video/background_segm.hpp>		

#define lScoreThresh 0.02
#define lScoreInit 0.14
		
struct face {
	cv::Rect roi;
	std::string id;
	int views;
};

class HandTracker
{
	public:
		HandTracker();
		~HandTracker();
				
	private:
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		ros::Publisher hand_face_pub;
				
		void callback(const sensor_msgs::ImageConstPtr& immsg, const facetracking::ROIArrayConstPtr& msg); // Detected face array/ image callback

		message_filters::TimeSynchronizer<sensor_msgs::Image, facetracking::ROIArray>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<facetracking::ROIArray> roi_sub;
		face face_found;
		
		
		cv::MatND hist1;
		cv::Ptr<cv::BackgroundSubtractor> pMOG2;
		cv::Mat fgMaskMOG2;
		
		double tempS[2];
		
		void checkHandsInitialisation (cv::Mat likelihood, cv::Mat image3, double xShift,cv::RotatedRect &roi, bool &track, double &tempScore);
		void updateHandPos (cv::Mat likelihood, cv::Mat image3, cv::RotatedRect &roi, bool &track, face &face_in, double &tempScore);
		
		void updateFaceInfo (const facetracking::ROIArrayConstPtr& msg);
		cv::Mat getHandLikelihood (cv::Mat input, face &face_in);
		void HandDetector (cv::Mat likelihood, face &face_in, cv::Mat image3);
		cv::Rect adjustRect (cv::Rect temp, cv::Size size);
		
		std::vector<cv::KalmanFilter> tracker;
		std::vector<cv::RotatedRect> box;
		bool tracked[2];
		
		handblobtracker::HFPose2DArray pfPose;
};

#endif

