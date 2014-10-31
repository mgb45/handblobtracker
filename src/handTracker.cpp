#include "handTracker.h"

using namespace cv;
using namespace std;

//TODO: Remove hard coded image dimensions, parametise properly.

// Constructer: Body tracker 
HandTracker::HandTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/likelihood",1); //ROS
	hand_face_pub = nh.advertise<measurementproposals::HFPose2DArray>("/faceHandPose", 10);
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	roi_sub.subscribe(nh, "/faceROIs", 1); // requires face array input
	
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, faceTracking::ROIArray>(image_sub,roi_sub,10);
	sync->registerCallback(boost::bind(&HandTracker::callback, this, _1, _2));
	
	face_found.views = 0;
		
	cv::Mat subImg1 = cv::Mat::zeros(50,50,CV_8UC3);
	
	int histSize[] = {35,35};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist1,2,histSize, rangesh, true, false);
	
	pMOG2 = new BackgroundSubtractorMOG2();
	  
}

HandTracker::~HandTracker()
{
	delete sync;
}

void HandTracker::checkHandsInitialisation(cv::Mat likelihood, cv::Mat image3)
{
	cv::RotatedRect roi;
	double bestScore = 0;
	/********Left right hand initialisation ************/
	cv::Rect temp;
	box.clear();
	score.clear();
	for (int i = 0; i <= (int)image3.cols-(int)image3.cols/8; i+=(int)image3.cols/8)
	{
		for (int j = 0; j <= (int)image3.rows-(int)image3.rows/8; j+=(int)image3.rows/8)
		{
			temp.x = i;
			temp.y = j;
			temp.width = (int)image3.cols/8;
			temp.height = (int)image3.rows/8;
			rectangle(image3, temp, Scalar(255,0,0), 2, 8, 0);
			roi = CamShift(likelihood, temp, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 5, 1 ));
			temp = roi.boundingRect();
			temp = adjustRect(temp,image3.size());
			bestScore = cv::sum(likelihood(temp))[0]/(255.0*M_PI*temp.width*temp.height/4.0);
						
			if (((bestScore > lScoreInit)||(temp.width >= 5)||(temp.height >= 5))&&((roi.center.x<i+temp.width)&&(roi.center.x>i)&&(roi.center.y<j+temp.height)&&(roi.center.y>j)))
			{
				box.push_back(roi);
				score.push_back(bestScore);
				try
				{
					ellipse(image3, roi, Scalar(255,0,0), 2, 8);
				}
				catch( cv::Exception& e )
				{
					const char* err_msg = e.what();
					ROS_ERROR("%s",err_msg);
				}	
			}
		}
	}
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

// Hand detector: given likelihood of skin, detects hands and intialising area and uses camshift to track them
void HandTracker::HandDetector(cv::Mat likelihood, face &face_in, cv::Mat image3)
{
	//Set face probability to zero, not a hand
	cv::Rect roi_enlarged; // enlarge face to cover neck and ear blobs
	roi_enlarged.height = face_in.roi.height*1.9;
	roi_enlarged.width = face_in.roi.width*1.5;
	roi_enlarged.x = face_in.roi.width/2 + face_in.roi.x - roi_enlarged.width/2;
	roi_enlarged.y = face_in.roi.height/2 + face_in.roi.y - roi_enlarged.height/3;
	roi_enlarged = adjustRect(roi_enlarged,image3.size());
		
	try
	{
		ellipse(likelihood, RotatedRect(Point2f(roi_enlarged.x+roi_enlarged.width/2.0,roi_enlarged.y+roi_enlarged.height/2.0),Size2f(roi_enlarged.width,roi_enlarged.height),0.0), Scalar(0,0,0), -1, 8);
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		ROS_ERROR("%s",err_msg);
	}	
	
	cvtColor(likelihood,image3,CV_GRAY2RGB);
	
	checkHandsInitialisation(likelihood,image3);
}

// Gets skin colour likelihood map from face using back projection in Lab
cv::Mat HandTracker::getHandLikelihood(cv::Mat input, face &face_in)
{
	cv::Mat image4;
	cvtColor(input,image4,CV_BGR2Lab);
				
	MatND hist;
	int histSize[] = {35,35};
	float h_range[] = {0, 255};
	float s_range[] = {0, 255};
	const float* rangesh[] = {h_range,s_range};
	
	cv::Rect rec_reduced;
	rec_reduced.x = face_in.roi.x+ face_in.roi.width/4;
	rec_reduced.y = face_in.roi.y+ face_in.roi.height/4;
	rec_reduced.width = face_in.roi.width - 2*face_in.roi.width/4;
	rec_reduced.height = face_in.roi.height- 2*face_in.roi.height/4;
	
	pMOG2->operator()(input,fgMaskMOG2,-10);
	
	// Generate output image
	cv::Mat foreground(image4.size(),CV_8UC3,cv::Scalar(255,255,255)); // all white image
	image4.copyTo(foreground,fgMaskMOG2); // bg pixels not copied
	
	cv::Mat subImg1 = image4(rec_reduced);
	
	int channels[] = {1, 2};
	calcHist(&subImg1,1,channels,Mat(),hist,2,histSize, rangesh, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	hist1 = 0.95*hist1 + 0.05*hist;
	
	cv::Mat temp1(input.rows,input.cols,CV_64F);
	calcBackProject(&foreground,1,channels,hist1,temp1,rangesh, 1, true);
	
	Mat element0 = getStructuringElement(MORPH_ELLIPSE, Size(7,7), Point(3,3));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(11,11), Point(5,5));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
	
	dilate(temp1,temp1,element0);
	erode(temp1,temp1,element1);
	dilate(temp1,temp1,element2);

	cv::Mat image3 = cv::Mat::zeros(image4.rows,image4.cols,CV_8UC3);
	
	// Detect hands and update pose estimate
	HandDetector(temp1,face_in,image3);
				
	// Draw face rectangles on display image
	try
	{
		rectangle(image3, Point(face_in.roi.x,face_in.roi.y), Point(face_in.roi.x+face_in.roi.width,face_in.roi.y+face_in.roi.height), Scalar(255,255,255), 4, 8, 0);
		rectangle(image3, Point(rec_reduced.x,rec_reduced.y), Point(rec_reduced.x+rec_reduced.width,rec_reduced.y+rec_reduced.height), Scalar(0,255,0), 4, 8, 0);
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		ROS_ERROR("%s",err_msg);
	}	
		
	return image3;
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
		img2.encoding = "rgb8";
		img2.header = immsg->header;
		img2.image = outputImage;			
		pub.publish(img2.toImageMsg()); // publish result image
		
		measurementproposals::HFPose2D rosHands;
		measurementproposals::HFPose2DArray rosHandsArr;
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + int(face_found.roi.height/2.0);
		rosHandsArr.measurements.push_back(rosHands);
		rosHands.x = face_found.roi.x + int(face_found.roi.width/2.0);
		rosHands.y = face_found.roi.y + 3.25/2.0*face_found.roi.height;
		rosHandsArr.measurements.push_back(rosHands); //Neck
		for (int i = 0; i < (int)box.size(); i++)
		{
			rosHands.x = box[i].center.x;
			rosHands.y = box[i].center.y;
			rosHandsArr.measurements.push_back(rosHands);
		}
		rosHandsArr.header = msg->header;
		rosHandsArr.id = face_found.id;
		hand_face_pub.publish(rosHandsArr);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}		
}
