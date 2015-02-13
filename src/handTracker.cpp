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
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
		
	cv::Mat subImg1 = cv::Mat(10,10,CV_8UC3);
	randu(subImg1,0,255);
	
	int histSize[] = {40,40};
	float r_range[] = {0, 255};
	float g_range[] = {0, 255};
	float b_range[] = {0, 255};
	const float* ranges[] = {r_range,g_range};
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, ranges, true, false);
	
	pMOG2 = new BackgroundSubtractorMOG2(30,15,false);
		  
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
	//roi_enlarged.height = face_in.roi.height/1.6;
	//roi_enlarged.width = face_in.roi.width/2;
	//roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	//roi_enlarged.y = face_in.roi.y;// - roi_enlarged.height/5;
	//roi_enlarged = adjustRect(roi_enlarged,temp1.size());
	rec_reduced.x = face_in.roi.width/2 + face_in.roi.x - face_in.roi.width/8;
	rec_reduced.y = face_in.roi.y + 0.25*face_in.roi.height - face_in.roi.height/8;
	rec_reduced.width = 2*face_in.roi.width/8;
	rec_reduced.height = 2*face_in.roi.height/8;
	
	
	//pMOG2->operator()(input,fgMaskMOG2,-10);
	cv::Mat fgMaskMOG;
	pMOG2->operator()(input,fgMaskMOG,0.0005);
	//GaussianBlur(fgMaskMOG, fgMaskMOG, cv::Size(7,7), 3, 93, BORDER_DEFAULT);
	
	// Generate output image
	cv::Mat foreground; // all white image
	image4.copyTo(foreground,fgMaskMOG); // bg pixels not copied
	//image4.copyTo(foreground); // bg pixels copied
	
	cv::Mat subImg1 = image4(rec_reduced);
	
	MatND hist;
	int histSize[] = {20,20,20};
	float r_range[] = {0, 255};
	float g_range[] = {0, 255};
	float b_range[] = {0, 255};
	const float* ranges[] = {r_range,g_range,b_range};
	int channels[] = {0,1,2};
	calcHist(&subImg1,1,channels,Mat(),hist,3,histSize, ranges, true, false);
	//normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	//hist1 = hist1 + hist;
	
	cv::Mat temp1;
	calcBackProject(&foreground,1,channels,hist,temp1,ranges, 1, true);
	normalize(temp1, temp1, 0, 255, NORM_MINMAX, -1, Mat());
	//temp1 = temp1;
	
	//Mat element0 = getStructuringElement(MORPH_ELLIPSE, Size(15,15), Point(9,9));
	////Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(11,11), Point(5,5));
	//Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(7,7), Point(2,2));
	//Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(4,4), Point(2,2));
	
	////erode(fgMaskMOG,fgMaskMOG,element3);
	////dilate(fgMaskMOG,fgMaskMOG,element0);
	////~ erode(fgMaskMOG,fgMaskMOG,element1);
	////~ dilate(fgMaskMOG,fgMaskMOG,element2);
	//dilate(temp1,temp1,element3);
	//erode(temp1,temp1,element2);	
	
	//Set face probability to zero, not a hand
	cv::Rect roi_enlarged; // enlarge face to cover neck and ear blobs
	roi_enlarged.height = face_in.roi.height/1.5;
	roi_enlarged.width = face_in.roi.width/2;
	roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	roi_enlarged.y = face_in.roi.y;// - roi_enlarged.height/5;
	roi_enlarged = adjustRect(roi_enlarged,temp1.size());
		
	try
	{
		ellipse(temp1, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);
		//ellipse(temp1, RotatedRect(Point2f(rec_reduced.x+rec_reduced.width/2.0,rec_reduced.y+rec_reduced.height/2.0),Size2f(rec_reduced.width,rec_reduced.height),0.0), Scalar(255,0,0), -1, 8);
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		ROS_ERROR("%s",err_msg);
	}	
		
	return temp1;
}

// Update tracked face with latest info
void HandTracker::updateFaceInfo(const faceTracking::ROIArrayConstPtr& msg)
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
void HandTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const faceTracking::ROIArrayConstPtr& msg)
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
