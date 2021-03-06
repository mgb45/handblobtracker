#include "handTracker.h"

using namespace cv;
using namespace std;

//TODO: Remove hard coded image dimensions, parametise properly.

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/likelihood",1); //ROS
			
	image_sub.subscribe(nh, "/rgb/image_color", 15); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 10); // requires face array input
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, facetracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
		
	cv::Mat subImg1 = cv::Mat::zeros(50,50,CV_8UC3);
	randu(subImg1,0,255);
	
	int histSize[] = {10,10};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	float v_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	int channels[] = {1,2};
	calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, rangesh, true, false);
	
	pMOG2 = new BackgroundSubtractorMOG2(55,5,true);
	  
}

HandTracker::~HandTracker()
{
	delete sync;
}

cv::Rect HandTracker::adjustRect(cv::Rect temp,cv::Size size)
{
	cv::Rect newRect;
	newRect.x = min(max(temp.x,1),size.width-5);
	newRect.y = min(max(temp.y,1),size.height-5);
	newRect.width = min(temp.width,size.width-newRect.x);
	newRect.height = min(temp.height,size.height-newRect.y);
	return newRect;
}

// Gets skin colour likelihood map from face using back projection in Lab
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image4;
	cvtColor(input,image4,CV_BGR2Lab);
				
	cv::Rect rec_reduced;
	rec_reduced.x = face_in.roi.x+ face_in.roi.width/3;
	rec_reduced.y = face_in.roi.y+ face_in.roi.height/3;
	rec_reduced.width = face_in.roi.width/3;
	rec_reduced.height = face_in.roi.height/3;
	
	cv::Mat mask1 = cv::Mat::zeros(input.rows,input.cols,CV_8UC1);
	try
	{
		ellipse(mask1, RotatedRect(Point2f(face_in.roi.x+face_in.roi.width/2.0,input.rows/2.0),Size2f(face_in.roi.width*8,input.rows),0.0), Scalar(255,255,255), -1, 8);
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		ROS_ERROR("%s",err_msg);
	}	
		
	//image4 = image4&mask1;
	
	cv::Mat fgMaskMOG; // all white image
	//pMOG2->operator()(input,fgMaskMOG2,-10);
	
	pMOG2->operator()(input,fgMaskMOG,5e-3);
	
		//Mat element0 = getStructuringElement(MORPH_ELLIPSE, Size(7,7), Point(3,3));
	//Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(2,2));
	
	//erode(temp1,temp1,element1);
	//dilate(temp1,temp1,element0);
	dilate(fgMaskMOG,fgMaskMOG,element2);
	dilate(fgMaskMOG,fgMaskMOG,element2);
	
	// Generate output image
	cv::Mat foreground; // all white image
	
	//image4.copyTo(foreground,fgMaskMOG); // bg pixels not copied
	image4.copyTo(foreground,mask1&fgMaskMOG); // bg pixels copied
		
	cv::Mat subImg1 = image4(rec_reduced);
	
	MatND hist;
	int histSize[] = {10,10};
	float r_range[] = {0, 255};
	float g_range[] = {0, 255};
	float b_range[] = {0, 255};
	const float* ranges[] = {r_range,g_range};
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist,2,histSize, ranges, true, false);
	//normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	hist1 = 0.95*hist1 + 0.05*hist;
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	calcBackProject(&foreground,1,channels,hist1,temp1,ranges, 1, true);
	normalize(temp1, temp1, 0, 255, NORM_MINMAX, -1, Mat());
	
	medianBlur(temp1,temp1,15);
	
	//erode(temp1,temp1,element1);
	//dilate(temp1,temp1,element0);
	//dilate(fgMaskMOG,fgMaskMOG,element2);
	//dilate(fgMaskMOG,fgMaskMOG,element2);


	//Set face probability to zero, not a hand
	cv::Rect roi_enlarged; // enlarge face to cover neck and ear blobs
	roi_enlarged.height = face_in.roi.height*2.4;
	roi_enlarged.width = face_in.roi.width*1.8;
	roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	roi_enlarged.y = face_in.roi.height/2 + face_in.roi.y - roi_enlarged.height/3;
	roi_enlarged = adjustRect(roi_enlarged,temp1.size());
		
	try
	{
		ellipse(temp1, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);
		rectangle(temp1, Point(0,temp1.rows-100), Point(temp1.cols,temp1.rows), Scalar(0,0,0), -1, 8);
		
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		ROS_ERROR("%s",err_msg);
	}	
		
	return temp1;
}

// Update tracked face with latest info
void HandTracker::updateFaceInfo(const facetracking::ROIArrayConstPtr& msg)
{
	if (msg->ROIs.size() > 0)
	{
		if (face_found.views <= 0) // if no faces exist yet
		{
			//pick first face in list
			face_found.roi = cv::Rect(msg->ROIs[0].x_offset,msg->ROIs[0].y_offset,msg->ROIs[0].width,msg->ROIs[0].height);
			face_found.id = msg->ids[0];
			face_found.views = 1;
		}
		else
		{
			face_found.views--;
			for (int i = 0; i < (int)msg->ROIs.size(); i++) // Assume no duplicate faces in list, iterate through detected faces
			{
				if (face_found.id.compare(msg->ids[i]) == 0) // old face
				{
					face_found.roi = cv::Rect(msg->ROIs[i].x_offset,msg->ROIs[i].y_offset,msg->ROIs[i].width,msg->ROIs[i].height); //update roi
					face_found.id = msg->ids[i];
					face_found.views++;
				}
			}
		}
	}
	else
	{
		face_found.views = 0;
	}
}

// image and face roi callback
void HandTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const facetracking::ROIArrayConstPtr& msg)
{
	try
	{	
		cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
			
		updateFaceInfo(msg); // update face list
		
		cv::Mat outputImage = cv::Mat::zeros(image.rows,image.cols,image.type()); // display image
		outputImage = getHandLikelihood(image,face_found);

		cv_bridge::CvImage img2;
		img2.encoding = "mono8";
		img2.header = immsg->header;
		img2.image = outputImage;			
		pub.publish(img2.toImageMsg()); // publish result image
	
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}
