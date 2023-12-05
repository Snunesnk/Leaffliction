#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <random>

void ImageProcessing::Rotate(cv::Mat& image, double minDistr, double maxDistr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double angle = distr(gen);
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

void ImageProcessing::Blur(cv::Mat& image, double minDistr, double maxDistr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double sigma = distr(gen);
	cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void ImageProcessing::Contrast(cv::Mat& image, double minDistr, double maxDistr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double alpha = distr(gen);
	image.convertTo(image, -1, alpha, 0);
}

void ImageProcessing::Scale(cv::Mat& image, double minDistr, double maxDistr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double factor = distr(gen);
	cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, factor);
	// Apply zoom to the original image
	cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::Illumination(cv::Mat& image, double minDistr, double maxDistr) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double brightness = distr(gen);
	image += cv::Scalar(brightness, brightness, brightness);
}

void ImageProcessing::Projective(cv::Mat& image, double minDistr, double maxDistr) {
	// Randomizer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double topLY = distr(gen);
	double topRX = distr(gen);
	double botLX = distr(gen);
	double botRY = distr(gen);
	double botRX = distr(gen);
	// Source points
	std::vector<cv::Point2f> srcPoints;
	srcPoints.push_back(cv::Point2f(0, 0));                           // Top-left corner
	srcPoints.push_back(cv::Point2f(image.cols - 1, 0));              // Top-right corner
	srcPoints.push_back(cv::Point2f(0, image.rows - 1));              // Bottom-left corner
	srcPoints.push_back(cv::Point2f(image.cols - 1, image.rows - 1)); // Bottom-right corner
	// Destination points
	std::vector<cv::Point2f> dstPoints;
	dstPoints.push_back(cv::Point2f(0, topLY));
	dstPoints.push_back(cv::Point2f(image.cols - topRX, 0));
	dstPoints.push_back(cv::Point2f(botLX, image.rows - 1));
	dstPoints.push_back(cv::Point2f(image.cols - botRX, image.rows - botRY));
	// Get the Perspective Transform Matrix i.e. M
	cv::Mat warpMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
	// Apply the perspective transformation to the image
	cv::warpPerspective(image, image, warpMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::ConvertToGrayScale(cv::Mat& image) {
	// Créez une copie de l'image d'origine en couleur
	cv::Mat originalImage = image.clone();

	// Convertissez l'image en niveaux de gris
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// Copiez le canal de luminance (niveaux de gris) dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	image.copyTo(originalImage);

}

void ImageProcessing::ColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color) {
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

void ImageProcessing::EqualizeHistogram(cv::Mat& image) {
	// Créez une copie de l'image d'origine en couleur
	cv::Mat originalImage = image.clone();

	// Convertissez l'image en niveaux de gris
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// Égalisez l'histogramme en niveaux de gris
	cv::equalizeHist(image, image);

	// Copiez le canal de luminance égalisé dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	image.copyTo(originalImage, image);
}

void ImageProcessing::EqualizeHistogramColor(cv::Mat& image) {
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	for (int i = 0; i < channels.size(); i++) {
		cv::equalizeHist(channels[i], channels[i]);
	}
	cv::merge(channels, image);
}

void ImageProcessing::EqualizeHistogramSaturation(cv::Mat& image) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsvImage, channels);
	cv::equalizeHist(channels[1], channels[1]);
	cv::merge(channels, hsvImage);
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::EqualizeHistogramValue(cv::Mat& image) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsvImage, channels);
	cv::equalizeHist(channels[2], channels[2]);
	cv::merge(channels, hsvImage);
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::SimpleBinarization(cv::Mat& inputOutputImage, int threshold) {
	// Créez une copie de l'image d'origine en couleur
	cv::Mat originalImage = inputOutputImage.clone();

	// Convertissez l'image en niveaux de gris si ce n'est pas déjà fait
	if (inputOutputImage.channels() > 1) {
		cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_BGR2GRAY);
	}

	// Appliquez la binarisation en niveaux de gris
	cv::threshold(inputOutputImage, inputOutputImage, threshold, 255, cv::THRESH_BINARY);

	// Copiez le canal de luminance (niveaux de gris) binarisé dans les canaux R, G et B de l'image en couleur d'origine
	cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_GRAY2BGR);
	inputOutputImage.copyTo(originalImage);

}

void ImageProcessing::DetectORBKeyPoints(cv::Mat& image) {
	// Créer un détecteur ORB
	cv::Ptr<cv::ORB> orb = cv::ORB::create();

	// Détecter les points d'intérêt (keypoints)
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);

	// Créer un masque vide de la même taille que l'image avec 3 canaux
	//cv::Mat mask(image.size(), image.type());
	//mask.setTo(cv::Scalar(255, 255, 255));

	// Dessiner les keypoints en bleu sur le masque
	for (const cv::KeyPoint& kp : keypoints) {
		cv::Point2f pt = kp.pt;
		cv::circle(image, pt, 3, cv::Scalar(255, 0, 0), -1); // Dessiner un petit cercle bleu
	}
}

void ImageProcessing::ExtractHue(cv::Mat& image, double lower, double upper) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++) {
		for (int j = 0; j < hsvImage.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				double hue = pixel[0];
				if (hue >= lower && hue <= upper) {
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;
				}
			}
		}
	}
}

void ImageProcessing::ExtractSaturation(cv::Mat& image, double threshold) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++) {
		for (int j = 0; j < hsvImage.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				if (hsvpixel[1] <= threshold) {
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;
				}
			}
		}
	}
}

void ImageProcessing::ExtractValue(cv::Mat& image, double threshold) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < hsvImage.rows; i++) {
		for (int j = 0; j < hsvImage.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
				if (hsvpixel[2] >= threshold) {
					pixel[0] = 0;
					pixel[1] = 0;
					pixel[2] = 0;
				}
			}
		}
	}
}

void ImageProcessing::ExtractRedChannel(cv::Mat& image) {
	// Extraire le canal Rouge (Red)
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				pixel[0] = 0;
				pixel[1] = 0;
			}
		}
	}
}

void ImageProcessing::ExtractGreenChannel(cv::Mat& image) {
	// Extraire le canal Vert (Green)
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				pixel[0] = 0;
				pixel[2] = 0;
			}
		}
	}
}

void ImageProcessing::ExtractBlueChannel(cv::Mat& image) {
	// Extraire le canal Bleu (Blue)
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				pixel[1] = 0;
				pixel[2] = 0;
			}
		}
	}
}

void ImageProcessing::drawPolylinesAroundObject(cv::Mat image) {
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
		// Calculate the convex hull of all points
		std::vector<cv::Point> convexHullPoints;
		cv::convexHull(allPoints, convexHullPoints);

		// Draw the convex hull polygon
		cv::polylines(image, convexHullPoints, true, cv::Scalar(255, 0, 255), 2);
	}
}

void ImageProcessing::drawRectangleAroundObject(cv::Mat image) {
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
		// Calculate the minimum bounding rectangle around all points
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		// Get the four corner points of the rotated rectangle
		cv::Point2f rectPoints[4];
		minRect.points(rectPoints);

		// Draw the rotated rectangle around all objects
		for (int j = 0; j < 4; ++j) {
			cv::line(image, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(255, 0, 255), 2);
		}
	}
}

std::vector<cv::Point> ImageProcessing::getMinimumBoundingRectanglePoints(cv::Mat image) {
	std::vector<cv::Point> rectPoints;

	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
		// Calculate the minimum bounding rectangle around all points
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		// Get the four corner points of the rotated rectangle
		cv::Point2f rectPointsArray[4];
		minRect.points(rectPointsArray);

		// Convert the points to std::vector<cv::Point>
		for (int i = 0; i < 4; ++i) {
			rectPoints.push_back(rectPointsArray[i]);
		}
	}

	return rectPoints;
}

std::vector<cv::Point> ImageProcessing::getConvexHullPoints(cv::Mat image) {
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
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
		// Calculate the convex hull of all points
		cv::convexHull(allPoints, convexHullPoints);

		// Draw the convex hull points on the image
		for (size_t i = 0; i < convexHullPoints.size(); ++i) {
			cv::circle(image, convexHullPoints[i], 3, cv::Scalar(0, 255, 0), -1); // Draw a green circle at each point
		}
	}

	return convexHullPoints;
}

void ImageProcessing::cropImageWithPoints(cv::Mat& image, const std::vector<cv::Point>& points) {
	// Créez une image blanche de la même taille que l'image d'origine
	cv::Mat whiteBackground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

	// Copiez la région à l'intérieur des points de l'image d'origine dans l'image de fond blanc
	cv::Mat regionOfInterest(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
	std::vector<std::vector<cv::Point>> contourVector = { points };
	cv::drawContours(mask, contourVector, 0, cv::Scalar(255), cv::FILLED);
	image.copyTo(regionOfInterest, mask);

	// Remplacez la région d'origine par la région extraite sur fond blanc
	image = whiteBackground;
	regionOfInterest.copyTo(image, mask);
}

double ImageProcessing::calculateAspectRatioOfObjects(cv::Mat image) {
	// Create a mask for non-white pixels
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), mask); // Exclude pure white

	// Find the contours in the mask
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Combine all points from all contours into one vector
	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
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

std::vector<cv::Point> ImageProcessing::ExtractShape(cv::Mat& image) {

	cv::Mat clone = image.clone();
	ImageProcessing::EqualizeHistogramValue(clone);
	ImageProcessing::EqualizeHistogramSaturation(clone);
	ImageProcessing::ColorFiltering(clone, cv::Scalar(20, 50, 50), cv::Scalar(80, 255, 255));
	std::vector<cv::Point> points =  getConvexHullPoints(clone);

	clone = image.clone();
	ImageProcessing::EqualizeHistogramValue(clone);
	ImageProcessing::EqualizeHistogramSaturation(clone);
	ImageProcessing::cropImageWithPoints(clone, points);
	ImageProcessing::ColorFiltering(clone, cv::Scalar(5, 105, 15), cv::Scalar(80, 255, 255));
	cv::medianBlur(clone, clone, 31);

	cv::inRange(clone, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), clone); // Exclude pure white
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(clone, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	points = std::vector<cv::Point>();
	for (size_t i = 0; i < contours.size(); ++i) {
		points.insert(points.end(), contours[i].begin(), contours[i].end());
	}
	clone = image.clone();//
	ImageProcessing::cropImageWithPoints(clone, points);
	ImageProcessing::EqualizeHistogramValue(clone);
	ImageProcessing::EqualizeHistogramSaturation(clone);
	ImageProcessing::ColorFiltering(clone, cv::Scalar(0, 15, 5), cv::Scalar(180, 255, 255));
	cv::medianBlur(clone, clone, 21);
	cv::inRange(clone, cv::Scalar(0, 0, 0), cv::Scalar(254, 254, 254), clone); // Exclude pure white
	contours = std::vector<std::vector<cv::Point>>();
	hierarchy = std::vector<cv::Vec4i>();
	cv::findContours(clone, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	points = std::vector<cv::Point>();
	for (size_t i = 0; i < contours.size(); ++i) {
		points.insert(points.end(), contours[i].begin(), contours[i].end());
	}

	ImageProcessing::cropImageWithPoints(image, points);
	ImageProcessing::SimpleBinarization(image, 254);

	return points;
}