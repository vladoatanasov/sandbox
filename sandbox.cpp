#include <stdio.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void DisplayResult (Mat frame, Mat origFrame);
void DetectObject (Mat frame, Mat & destFrame, char *haarcascade);

int main (int argc, char **argv)
{
	int i = 0;
	char *haarcascade = NULL;
	char *file = NULL;

	if (argc == 1)
	{
		printf ("Usage: %s\n  [-hc <haar_cascade_file_name>]\n"
				"  [-img <image_file_name>] (optional)\n", argv[0]);

		return 0;
	}

	for (i = 1; i < argc; ++i)
	{
		if (!strcmp (argv[i], "-hc"))
		{
			haarcascade = argv[++i];
		}
		else if (!strcmp (argv[i], "-img"))
		{
			file = argv[++i];
		}
	}


	CvCapture *capture = 0;
	Mat frame, frameCopy, origFrame;

	if (file != NULL)
	{
		frame = imread (file, CV_LOAD_IMAGE_COLOR);
		//check image size and resize if nessesarry
		Size s = frame.size ();
		if (s.width > 800 || s.height > 800)
		{
			if (s.width > s.height)
			{
				double idx = (double) s.width / (double) s.height;
				Size newSize ((int) (800 * idx), 800);
				resize (frame, frame, newSize);
			}
			else
			{
				double idx = (double) s.height / (double) s.width;
				Size newSize (800, (int) (800 * idx));
				resize (frame, frame, newSize);
			}
		}

		frame.copyTo (origFrame);

		DetectObject (frame, frame, haarcascade);
		DisplayResult (frame, origFrame);

		imwrite ("output.jpg", frame);

		waitKey (0);

		return 0;
	}
	else
	{
		capture = cvCaptureFromCAM (0);	//0=default, -1=any camera, 1..99=your camera
		if (!capture)
		{
			printf ("No camera detected\n");
			return 0;
		}

		if (capture)
		{
			printf ("Capturing ...\n");
			for (;;)
			{
				IplImage *iplImg = cvQueryFrame (capture);
				frame = iplImg;
				if (frame.empty ())
					break;
				if (iplImg->origin == IPL_ORIGIN_TL)
					frame.copyTo (frameCopy);
				else
					flip (frame, frameCopy, 0);

				if (waitKey (10) >= 0)
					cvReleaseCapture (&capture);

				frame.copyTo (origFrame);

				//threshold(frame, frame, 100, 1000, 1);        
				DetectObject (frame, frame, haarcascade);

				DisplayResult (frame, origFrame);
			}

			waitKey (0);

			return 0;
		}
	}
}

/** @function DisplayResult */
void DisplayResult (Mat frame, Mat origFrame)
{
	imshow ("Processed", frame);
	imshow ("Source", origFrame);
}

/** @function detectObject */
void DetectObject (Mat frame, Mat & destFrame, char *haarcascade)
{
	CascadeClassifier object_cascade;

	std::vector < Rect > objects;
	Mat frame_gray;

	object_cascade.load (haarcascade);

	cvtColor (frame, frame_gray, CV_BGR2GRAY);
	equalizeHist (frame_gray, frame_gray);

	//-- Detect objects
	object_cascade.detectMultiScale (frame_gray, objects, 1.1, 2,
			0 | CV_HAAR_SCALE_IMAGE, Size (30, 30));

	for (size_t i = 0; i < objects.size (); i++)
	{
		Point center (objects[i].x + objects[i].width * 0.5,
				objects[i].y + objects[i].height * 0.5);
		//ellipse( frame, center, Size( objects[i].width*0.5, objects[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
		rectangle (frame, objects[i], Scalar (0, 255, 255), 2, CV_AA, 0);
	}

	frame.copyTo (destFrame);
}
