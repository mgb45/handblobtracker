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
#include "faceTracking/ROIArray.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sstream>
#include <string>
#include <ros/package.h>
#include "measurementproposals/HFPose2D.h"
#include "measurementproposals/HFPose2DArray.h"
#include <opencv2/video/background_segm.hpp>		

struct face {
	cv::Rect roi;
	std::string id;
	int views;
};

struct hand {
	cv::RotatedRect roi;
	double score;
	
	bool operator < (const hand& str) const
    {
        return (score > str.score);
    }
};

class HandTracker
{
	public:
		HandTracker();
		~HandTracker();
				
	private:
		ros::NodeHandle nh;
		image_transport::Publisher pub;
						
		void callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg); // Detected face array/ image callback

		message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<faceTracking::ROIArray> roi_sub;
		face face_found;
		
		
		cv::MatND hist1;
		cv::Ptr<cv::BackgroundSubtractor> pMOG2;
		cv::Mat fgMaskMOG2;
					
		void updateFaceInfo (const faceTracking::ROIArrayConstPtr& msg);
		cv::Mat getHandLikelihood (cv::Mat input, face &face_in);
		cv::Rect adjustRect(cv::Rect temp,cv::Size size);
	
};

#endif

