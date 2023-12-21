#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <random>

void ImageProcessing::Rotate(cv::Mat& image, double minDistr, double maxDistr)
{
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

void ImageProcessing::Distort(cv::Mat& image)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> freqDistr(0.01, 0.015); // Fréquence des ondes
	std::uniform_real_distribution<> ampDistr(5.0, 7.5);  // Amplitude des ondes

	double frequency = freqDistr(gen);
	double amplitude = ampDistr(gen);

	cv::Mat dst(image.size(), image.type(), cv::Scalar(255, 255, 255));

	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			int newY = y + static_cast<int>(amplitude * sin(x * frequency * 2 * 3.14));

			newY = std::min(std::max(newY, 0), image.rows - 1);

			dst.at<cv::Vec3b>(newY, x) = image.at<cv::Vec3b>(y, x);
		}
	}

	image = dst;
}

void ImageProcessing::Flip(cv::Mat& image)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distr(false, true);
	int horizontal = distr(gen);
	int flipCode = horizontal ? true : false;
	cv::flip(image, image, flipCode);
}

void ImageProcessing::Shear(cv::Mat& image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double shearAmount = distr(gen);

	// Calculer l'augmentation de la largeur due au cisaillement
	double increasedWidth = image.cols + std::abs(shearAmount) * image.rows;

	// Calculer le facteur d'échelle nécessaire
	double scaleFactor = static_cast<double>(image.cols) / increasedWidth;

	// Calculer le décalage pour centrer l'image
	double offsetX = (image.cols - (scaleFactor * increasedWidth)) / 2.0;
	double offsetY = (image.rows - (scaleFactor * image.rows)) / 2.0;

	// Ajuster la matrice pour le cisaillement, la mise à l'échelle et le décalage
	cv::Mat shearMatrix = (cv::Mat_<double>(2, 3) << scaleFactor, shearAmount * scaleFactor, offsetX, 0, scaleFactor, offsetY);

	// Appliquer la transformation
	cv::Mat dst;
	cv::warpAffine(image, dst, shearMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

	image = dst;
}

void ImageProcessing::Scale(cv::Mat& image, double minDistr, double maxDistr)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distr(minDistr, maxDistr);
	double factor = distr(gen);
	cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, factor);
	// Apply zoom to the original image
	cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::Projective(cv::Mat& image, float minDistr, float maxDistr)
{
	// Randomizer
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(minDistr, maxDistr);
	float topLY = distr(gen);
	float topRX = distr(gen);
	float botLX = distr(gen);
	float botRY = distr(gen);
	float botRX = distr(gen);
	// Source points
	std::vector<cv::Point2f> srcPoints;
	srcPoints.push_back(cv::Point2f(0, 0)); // Top-left corner
	srcPoints.push_back(cv::Point2f(static_cast<float>(image.cols) - 1, 0)); // Top-right corner
	srcPoints.push_back(cv::Point2f(0, static_cast<float>(image.rows) - 1)); // Bottom-left corner
	srcPoints.push_back(cv::Point2f(static_cast<float>(image.cols) - 1, static_cast<float>(image.rows) - 1)); // Bottom-right corner
	// Destination points
	std::vector<cv::Point2f> dstPoints;
	dstPoints.push_back(cv::Point2f(0, topLY));
	dstPoints.push_back(cv::Point2f(static_cast<float>(image.cols) - topRX, 0));
	dstPoints.push_back(cv::Point2f(botLX, static_cast<float>(image.rows) - 1));
	dstPoints.push_back(cv::Point2f(static_cast<float>(image.cols) - botRX, static_cast<float>(image.rows) - botRY));
	// Get the Perspective Transform Matrix i.e. M
	cv::Mat warpMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
	// Apply the perspective transformation to the image
	cv::warpPerspective(image, image, warpMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}


void ImageProcessing::ConvertToGray(cv::Mat& inputImage)
{
	cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
}

void ImageProcessing::BinarizeImage(cv::Mat& inputImage)
{
	cv::Mat grayImage;
	ImageProcessing::ConvertToGray(inputImage);
	cv::threshold(inputImage, inputImage, 128, 255, cv::THRESH_BINARY);
}

void ImageProcessing::ExtractYChannel(cv::Mat& inputImage)
{
	if (inputImage.channels() == 3) {
		cv::Mat yChannel;
		cv::cvtColor(inputImage, yChannel, cv::COLOR_BGR2YCrCb);
		std::vector<cv::Mat> channels;
		cv::split(yChannel, channels);
		inputImage = channels[0];
	}
}

void ImageProcessing::ApplyCannyEdgeDetection(cv::Mat& inputImage)
{
	cv::Mat grayImage;
	ImageProcessing::ConvertToGray(inputImage);
	cv::Canny(inputImage, inputImage, 100, 200);
}

void ImageProcessing::ApplyGaussianBlur(cv::Mat& inputImage, int kernelSize)
{
	cv::GaussianBlur(inputImage, inputImage, cv::Size(kernelSize, kernelSize), 0);
}

void ImageProcessing::ApplyContrastEnhancement(cv::Mat& inputImage, double factor)
{
	inputImage.convertTo(inputImage, -1, factor, 0);
}

void ImageProcessing::ColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	cv::Mat objectMask;
	cv::inRange(hsvImage, lowerBound, upperBound, objectMask);
	cv::Mat bgMask;
	cv::inRange(hsvImage, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 1), bgMask);
	cv::Mat invertedObjectMask;
	cv::bitwise_not(objectMask, invertedObjectMask);
	cv::Mat maskToChange;
	cv::bitwise_and(invertedObjectMask, ~bgMask, maskToChange);
	image.setTo(color, maskToChange);
}

void ImageProcessing::EqualizeHistogram(cv::Mat& image)
{
	cv::Mat originalImage = image.clone();
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(image, image);
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	image.copyTo(originalImage, image);
}

void ImageProcessing::EqualizeHistogramColor(cv::Mat& image)
{
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	for (int i = 0; i < channels.size(); i++) {
		cv::equalizeHist(channels[i], channels[i]);
	}
	cv::merge(channels, image);
}

void ImageProcessing::EqualizeHistogramSaturation(cv::Mat& image)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsvImage, channels);
	cv::equalizeHist(channels[1], channels[1]);
	cv::merge(channels, hsvImage);
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::EqualizeHistogramValue(cv::Mat& image)
{
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsvImage, channels);
	cv::equalizeHist(channels[2], channels[2]);
	cv::merge(channels, hsvImage);
	cv::cvtColor(hsvImage, image, cv::COLOR_HSV2BGR);
}

void ImageProcessing::DetectORBKeyPoints(cv::Mat& image)
{
	// Créer un détecteur ORB
	cv::Ptr<cv::ORB> orb = cv::ORB::create();

	// Détecter les points d'intérêt (keypoints)
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);

	for (const cv::KeyPoint& kp : keypoints) {
		cv::Point2f pt = kp.pt;
		cv::circle(image, pt, 3, cv::Scalar(255, 0, 0), -1);
	}
}

double ImageProcessing::calculateAspectRatioOfObjects(cv::Mat image)
{
	cv::Mat mask;
	cv::inRange(image, cv::Scalar(1, 1, 1), cv::Scalar(255, 255, 255), mask);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> allPoints;
	for (size_t i = 0; i < contours.size(); ++i) {
		allPoints.insert(allPoints.end(), contours[i].begin(), contours[i].end());
	}

	if (!allPoints.empty()) {
		cv::RotatedRect minRect = cv::minAreaRect(allPoints);

		double width = minRect.size.width;
		double height = minRect.size.height;

		double aspectRatio = (width > height) ? (width / height) : (height / width);

		return aspectRatio;
	}

	return 0.0;
}
std::vector<cv::Point> ImageProcessing::GetConvexHullPoints(cv::Mat image)
{
	std::vector<cv::Point> convexHullPoints;

	cv::Mat mask;
	cv::inRange(image, cv::Scalar(0, 0, 0), cv::Scalar(1, 1, 1), mask);
	
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(~mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<cv::Point> allPoints;
	for (const auto& contour : contours) {
		allPoints.insert(allPoints.end(), contour.begin(), contour.end());
	}

	if (!allPoints.empty()) {
		cv::convexHull(cv::Mat(allPoints), convexHullPoints);
	}

	return convexHullPoints;
}

void ImageProcessing::CropImageWithPoints(cv::Mat& image, const std::vector<cv::Point>& points)
{
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

	std::vector<std::vector<cv::Point>> contourVector = { points };
	cv::drawContours(mask, contourVector, 0, cv::Scalar(255), cv::FILLED);

	cv::Mat croppedImage = cv::Mat::zeros(image.size(), image.type());
	image.copyTo(croppedImage, mask);

	image = croppedImage;
}

void ImageProcessing::CutLeaf(cv::Mat& image)
{
	cv::Mat originalImage = image.clone();
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	for (int i = 0; i < originalImage.rows; i++) {
		for (int j = 0; j < originalImage.cols; j++) {
			cv::Vec3b& BGR = image.at<cv::Vec3b>(i, j);
			double value = BGR[0];
			if (BGR[2] < value) {
				value = BGR[2];
			}
			BGR[1] = (BGR[1] < value ? 0 : BGR[1] - value);
			BGR[0] = (BGR[0] < value ? 0 : BGR[0] - value);
			BGR[2] = (BGR[2] < value ? 0 : BGR[2] - value);

			if (hsvImage.at<cv::Vec3b>(i, j)[2] < 10) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
			if (BGR[2] >= BGR[1] - 5 && BGR[2] >= BGR[0] - 5 && abs(BGR[1] - BGR[0]) < 20) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
			if (BGR[0] >= BGR[1] - 5 && BGR[0] >= BGR[2] - 5 && abs(BGR[1] - BGR[2]) < 20) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
		}
	}
	std::vector<cv::Point> points = ImageProcessing::GetConvexHullPoints(image);
	image = originalImage.clone();
	ImageProcessing::CropImageWithPoints(image, points);

	//cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);	

	for (int i = 0; i < originalImage.rows; i++) {
		for (int j = 0; j < originalImage.cols; j++) {
			cv::Vec3b& BGR = image.at<cv::Vec3b>(i, j);

			uchar value = BGR[0];
			if (BGR[2] < value) {
				value = BGR[2];
			}

			BGR[1] = (BGR[1] < value ? 0 : BGR[1] - value);
			BGR[0] = (BGR[0] < value ? 0 : BGR[0] - value);
			BGR[2] = (BGR[2] < value ? 0 : BGR[2] - value);


			if (hsvImage.at<cv::Vec3b>(i, j)[2] < 10) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
			if (BGR[2] >= BGR[1] - 2 && BGR[2] >= BGR[0] - 2 && abs(BGR[1] - BGR[0]) < 15) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
			if (BGR[0] >= BGR[1] - 2 && BGR[0] >= BGR[2] - 2 && abs(BGR[1] - BGR[2]) < 20) {
				BGR[2] = 0;
				BGR[1] = 0;
				BGR[0] = 0;
			}
		}
	}
	
	cv::Mat grayscaleImage;
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(grayscaleImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::Mat mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
	cv::fillPoly(mask, contours, cv::Scalar(255));
	originalImage.copyTo(image, mask);


	int erosionSize = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1));
	cv::erode(image, image, element, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
	cv::findContours(grayscaleImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
	cv::fillPoly(mask, contours, cv::Scalar(255));
	originalImage.copyTo(image, mask);

	// Trouver le contour avec la plus grande aire
	double maxArea = 0.0;
	std::vector<cv::Point> maxContour;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > maxArea) {
			maxArea = area;
			maxContour = contour;
		}
	}

	// Calculer le rectangle englobant pour le contour maximal
	cv::Rect boundingBox = cv::boundingRect(maxContour);

	// Calculer le facteur de mise à l'échelle en conservant le rapport d'aspect
	double scale = std::min((double)image.cols / boundingBox.width,
		(double)image.rows / boundingBox.height);

	// Extraire et redimensionner la région de la feuille
	cv::Mat leafRegion = image(boundingBox);
	cv::Mat resizedLeaf;
	cv::resize(leafRegion, resizedLeaf, cv::Size(), scale, scale, cv::INTER_AREA);

	// Créer une nouvelle image avec un fond noir
	cv::Mat newImage(image.size(), image.type(), cv::Scalar::all(0));

	// Calculer la position pour centrer l'image redimensionnée
	cv::Rect roi((newImage.cols - resizedLeaf.cols) / 2,
		(newImage.rows - resizedLeaf.rows) / 2,
		resizedLeaf.cols, resizedLeaf.rows);

	// Placer l'image redimensionnée sur le fond noir
	resizedLeaf.copyTo(newImage(roi));

	// Mettre à jour l'image originale
	image = newImage.clone();
}
