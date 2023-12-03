#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <random>

void ImageProcessing::Rotate(cv::Mat &image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int angle = distr(gen);
	// Calculate the size of the resulting image after rotation
	cv::Rect boundingRect = cv::RotatedRect(cv::Point2f(image.cols / 2.0, image.rows / 2.0), image.size(), angle).boundingRect();
	// Automatically calculate the scaling factor
	double scale_factor = std::min(static_cast<double>(image.cols) / boundingRect.width, static_cast<double>(image.rows) / boundingRect.height);
	// Calculate the rotation matrix with scaling
	cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(image.cols / 2.0, image.rows / 2.0), angle, scale_factor);
	// Create a new image with a white background of the size of the original image
	cv::Mat rotatedImage = cv::Mat::zeros(image.size(), image.type());
	// Apply rotation (and scaling) to the original image
	cv::warpAffine(image, rotatedImage, rotationMatrix, rotatedImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	// Replace the original image with the rotated and scaled image
	image = rotatedImage;
}

void ImageProcessing::Blur(cv::Mat &image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int sigma = distr(gen);
	cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void ImageProcessing::Contrast(cv::Mat &image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int alpha = distr(gen);
	image.convertTo(image, -1, alpha, 0);
}

void ImageProcessing::Scale(cv::Mat &image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int factor = distr(gen);
	cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, factor);
	// Apply zoom to the original image
	cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::Illumination(cv::Mat &image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int brightness = distr(gen);
	image += cv::Scalar(brightness, brightness, brightness);
}

void ImageProcessing::Projective(cv::Mat &image, double minDistr, double maxDistr)
{
	// Randomizer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(minDistr, maxDistr);
	int topLY = distr(gen);
	int topRX = distr(gen);
	int botLX = distr(gen);
	int botRY = distr(gen);
	int botRX = distr(gen);

	std::vector<cv::Point2f> srcPoints;
	srcPoints.push_back(cv::Point2f(0, 0));							  // Top-left corner
	srcPoints.push_back(cv::Point2f(image.cols - 1, 0));			  // Top-right corner
	srcPoints.push_back(cv::Point2f(0, image.rows - 1));			  // Bottom-left corner
	srcPoints.push_back(cv::Point2f(image.cols - 1, image.rows - 1)); // Bottom-right corner

	std::vector<cv::Point2f> dstPoints;
	dstPoints.push_back(cv::Point2f(0, topLY));
	dstPoints.push_back(cv::Point2f(image.cols - topRX, 0));
	dstPoints.push_back(cv::Point2f(botLX, image.rows - 1));
	dstPoints.push_back(cv::Point2f(image.cols - botRX, image.rows - botRY));

	cv::Mat warpMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

	cv::warpPerspective(image, image, warpMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::ConvertToGrayScale(cv::Mat &image)
{
	// Cr�ez une copie de l'image d'origine en couleur
	cv::Mat originalImage = image.clone();

	// Convertissez l'image en niveaux de gris
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// Copiez le canal de luminance (niveaux de gris) dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	image.copyTo(originalImage);
}

void ImageProcessing::ColorFiltering(cv::Mat &image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color)
{
	// Convert the image to the HSV color space
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	// Create a mask for the specified color range
	cv::Mat colorMask;
	cv::inRange(hsvImage, lowerBound, upperBound, colorMask);

	cv::bitwise_not(colorMask, colorMask);
	// Set the color for the parts of the image that match the mask
	image.setTo(color, colorMask);
}

void ImageProcessing::EqualizeHistogram(cv::Mat &image)
{
	// Cr�ez une copie de l'image d'origine en couleur
	cv::Mat originalImage = image.clone();

	// Convertissez l'image en niveaux de gris
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// �galisez l'histogramme en niveaux de gris
	cv::equalizeHist(image, image);

	// Copiez le canal de luminance �galis� dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	image.copyTo(originalImage, image);
}

void ImageProcessing::SimpleBinarization(cv::Mat &inputOutputImage, int threshold)
{
	// Cr�ez une copie de l'image d'origine en couleur
	cv::Mat originalImage = inputOutputImage.clone();

	// Convertissez l'image en niveaux de gris si ce n'est pas d�j� fait
	if (inputOutputImage.channels() > 1)
	{
		cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_BGR2GRAY);
	}

	// Appliquez la binarisation en niveaux de gris
	cv::threshold(inputOutputImage, inputOutputImage, threshold, 255, cv::THRESH_BINARY);

	// Copiez le canal de luminance (niveaux de gris) binaris� dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_GRAY2BGR);
	inputOutputImage.copyTo(originalImage);
}

void ImageProcessing::DetectORBKeyPoints(cv::Mat &image)
{
	// Cr�er un d�tecteur ORB
	cv::Ptr<cv::ORB> orb = cv::ORB::create();

	// D�tecter les points d'int�r�t (keypoints)
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);

	// Cr�er un masque vide de la m�me taille que l'image avec 3 canaux
	// cv::Mat mask(image.size(), image.type());
	// mask.setTo(cv::Scalar(255, 255, 255));

	// Dessiner les keypoints en bleu sur le masque
	for (const cv::KeyPoint &kp : keypoints)
	{
		cv::Point2f pt = kp.pt;
		cv::circle(image, pt, 3, cv::Scalar(255, 0, 0), -1); // Dessiner un petit cercle bleu
	}
}

void ImageProcessing::ExtractHue(cv::Mat &image, double lower, double upper)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				cv::Vec3b &hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				double hue = pixel[0];
				if (hue > lower && hue < upper)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				}
			}
		}
	}
}

void ImageProcessing::ExtractSaturation(cv::Mat &image, double threshold)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				cv::Vec3b &hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				if (hsvpixel[1] < threshold)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				}
			}
		}
	}
}

void ImageProcessing::ExtractValue(cv::Mat &image, double threshold)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				cv::Vec3b &hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				if (hsvpixel[2] < threshold)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				}
			}
		}
	}
}

void ImageProcessing::ExtractRedChannel(cv::Mat &image)
{
	// Extraire le canal Rouge (Red)
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				pixel[0] = 255;
				pixel[1] = 255;
				pixel[2] = 255 - pixel[2];
			}
		}
	}
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = hsvImage.at<cv::Vec3b>(i, j);
			pixel[0] = (pixel[0] + 90) % 180;
		}
	}
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::ExtractGreenChannel(cv::Mat &image)
{
	// Extraire le canal Vert (Green)
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				pixel[0] = 255;
				pixel[1] = 255 - pixel[1];
				pixel[2] = 255;
			}
		}
	}
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = hsvImage.at<cv::Vec3b>(i, j);
			pixel[0] = (pixel[0] + 90) % 180;
		}
	}
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::ExtractBlueChannel(cv::Mat &image)
{
	// Extraire le canal Bleu (Blue)
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			cv::Vec3b &pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3)
			{
				pixel[0] = 255 - pixel[0];
				pixel[1] = 255;
				pixel[2] = 255;
			}
		}
	}
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++)
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			cv::Vec3b &pixel = hsvImage.at<cv::Vec3b>(i, j);
			pixel[0] = (pixel[0] + 90) % 180;
		}
	}
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::drawPolylinesAroundObject(cv::Mat image)
{
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty())
	{
		// Calculate the convex hull of all points
		std::vector<cv::Point> convexHullPoints;
		cv::convexHull(allPoints, convexHullPoints);

		// Draw the convex hull polygon
		cv::polylines(image, convexHullPoints, true, cv::Scalar(255, 0, 255), 2);
	}
}

void ImageProcessing::drawRectangleAroundObject(cv::Mat image)
{
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty())
	{
		// Calculate the minimum bounding rectangle around all points
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		// Get the four corner points of the rotated rectangle
		cv::Point2f rectPoints[4];
		minRect.points(rectPoints);

		// Draw the rotated rectangle around all objects
		for (int j = 0; j < 4; ++j)
		{
			cv::line(image, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(255, 0, 255), 2);
		}
	}
}

void ImageProcessing::drawAndScaleRectangleFromPoints(cv::Mat &image, std::vector<cv::Point> &parallelogramPoints)
{
	// V�rifie si parallelogramPoints contient au moins quatre points
	if (parallelogramPoints.size() >= 5)
	{
		// Trouver le centre du parall�logramme
		cv::Point center(0, 0);
		for (const cv::Point &point : parallelogramPoints)
		{
			center.x += point.x;
			center.y += point.y;
		}
		center.x /= parallelogramPoints.size();
		center.y /= parallelogramPoints.size();

		// Trier les points par angle par rapport au centre
		std::sort(parallelogramPoints.begin(), parallelogramPoints.end(), [&center](const cv::Point &a, const cv::Point &b)
				  {
			double angleA = atan2(a.y - center.y, a.x - center.x);
			double angleB = atan2(b.y - center.y, b.x - center.x);
			return angleA < angleB; });

		// Cr�ez une image vide pour le r�sultat
		cv::Mat result(image.size(), image.type(), cv::Scalar(255, 255, 255));

		// D�finissez les points source (les coins de l'image)
		std::vector<cv::Point2f> srcPoints(4);
		srcPoints[0] = cv::Point2f(0, 0);
		srcPoints[1] = cv::Point2f(image.cols - 1, 0);
		srcPoints[2] = cv::Point2f(image.cols - 1, image.rows - 1);
		srcPoints[3] = cv::Point2f(0, image.rows - 1);

		// Convertissez parallelogramPoints en un tableau de points de destination
		std::vector<cv::Point2f> dstPoints(4);
		for (int i = 0; i < 4; ++i)
		{
			dstPoints[i] = cv::Point2f(parallelogramPoints[i].x, parallelogramPoints[i].y);
		}

		// Calculez la transformation perspective � partir des points source et de destination
		cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

		// Appliquez la transformation perspective pour ajuster l'image au parall�logramme
		cv::warpPerspective(image, result, perspectiveMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

		// Remplacez l'image d'origine par l'image r�sultante
		image = result.clone();
	}

	// Dessinez le parall�logramme sur l'image
	for (int i = 0; i < parallelogramPoints.size(); ++i)
	{
		cv::line(image, parallelogramPoints[i], parallelogramPoints[(i + 1) % parallelogramPoints.size()], cv::Scalar(255, 0, 255), 2);
	}
}

std::vector<cv::Point> ImageProcessing::getMinimumBoundingRectanglePoints(cv::Mat image)
{
	std::vector<cv::Point> rectPoints;

	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty())
	{
		// Calculate the minimum bounding rectangle around all points
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		// Get the four corner points of the rotated rectangle
		cv::Point2f rectPointsArray[4];
		minRect.points(rectPointsArray);

		// Convert the points to std::vector<cv::Point>
		for (int i = 0; i < 4; ++i)
		{
			rectPoints.push_back(rectPointsArray[i]);
		}
	}

	return rectPoints;
}

std::vector<cv::Point> ImageProcessing::getConvexHullPoints(cv::Mat image)
{
	std::vector<cv::Point> convexHullPoints;

	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty())
	{
		// Calculate the convex hull of all points
		cv::convexHull(allPoints, convexHullPoints);

		// Draw the convex hull points on the image
		for (size_t i = 0; i < convexHullPoints.size(); ++i)
		{
			cv::circle(image, convexHullPoints[i], 3, cv::Scalar(0, 255, 0), -1); // Draw a green circle at each point
		}
	}

	return convexHullPoints;
}

void ImageProcessing::cropImageWithPoints(cv::Mat &image, const std::vector<cv::Point> &points)
{
	// Cr�ez une image blanche de la m�me taille que l'image d'origine
	cv::Mat whiteBackground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

	// Copiez la r�gion � l'int�rieur des points de l'image d'origine dans l'image de fond blanc
	cv::Mat regionOfInterest(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
	std::vector<std::vector<cv::Point>> contourVector = {points};
	cv::drawContours(mask, contourVector, 0, cv::Scalar(255), cv::FILLED);
	image.copyTo(regionOfInterest, mask);

	// Remplacez la r�gion d'origine par la r�gion extraite sur fond blanc
	image = whiteBackground;
	regionOfInterest.copyTo(image, mask);
}

double ImageProcessing::calculateAspectRatioOfObjects(cv::Mat image)
{
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty())
	{
		// Calculate the minimum bounding rectangle around all points
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		// Get the width and height of the rotated rectangle
		double width = minRect.size.width;
		double height = minRect.size.height;

		// Calculate the aspect ratio
		double aspectRatio = (width > height) ? (width / height) : (height / width);

		return aspectRatio;
	}

	// Return a default value if no objects are found
	return 0.0; // You can choose any default value here
}
