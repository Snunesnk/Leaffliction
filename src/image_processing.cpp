#include "image_processing.h"

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
	std::uniform_real_distribution<> freqDistr(0.01, 0.015);
	std::uniform_real_distribution<> ampDistr(5.0, 7.5);
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
	// Calculate the increased width due to shear
	double increasedWidth = image.cols + std::abs(shearAmount) * image.rows;
	// Calculate the scale factor needed
	double scaleFactor = static_cast<double>(image.cols) / increasedWidth;
	// Calculate the offset to center the image
	double offsetX = (image.cols - (scaleFactor * increasedWidth)) / 2.0;
	double offsetY = (image.rows - (scaleFactor * image.rows)) / 2.0;
	// Create a shear matrix for the transformation
	cv::Mat shearMatrix = (cv::Mat_<double>(2, 3) << scaleFactor, shearAmount * scaleFactor, offsetX, 0, scaleFactor, offsetY);
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
	// ORB detector
	cv::Ptr<cv::ORB> orb = cv::ORB::create();
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);
	for (const cv::KeyPoint& kp : keypoints) {
		cv::Point2f pt = kp.pt;
		cv::circle(image, pt, 3, cv::Scalar(255, 0, 0), -1);
	}
}

void ImageProcessing::ExtractLeafAndRescale(cv::Mat& image)
{
	cv::Mat originalImage = image.clone();
	cv::Mat hsvImage;
	// Cut
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
	image = originalImage.clone();
	// Get convexhull points and crop
	std::vector<cv::Point> convexHullPoints;
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<cv::Point> allPoints;
	for (const auto& contour : contours) {
		allPoints.insert(allPoints.end(), contour.begin(), contour.end());
	}
	if (!allPoints.empty()) {
		cv::convexHull(cv::Mat(allPoints), convexHullPoints);
		// Crop
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		std::vector<std::vector<cv::Point>> contourVector = { convexHullPoints };
		cv::drawContours(mask, contourVector, 0, cv::Scalar(255), cv::FILLED);
		cv::Mat croppedImage = cv::Mat::zeros(image.size(), image.type());
		image.copyTo(croppedImage, mask);
		image = croppedImage;
	}
	else {
		image = originalImage.clone();
	}
	// Cut
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
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	contours = std::vector<std::vector<cv::Point>>();
	cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::Mat mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
	cv::fillPoly(mask, contours, cv::Scalar(255));
	originalImage.copyTo(image, mask);
	// Erode
	int erosionSize = 9;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosionSize + 1, 2 * erosionSize + 1));
	cv::erode(image, image, element, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
	cv::fillPoly(mask, contours, cv::Scalar(255));
	originalImage.copyTo(image, mask);
	// Resize
	double maxArea = 0.0;
	std::vector<cv::Point> maxContour;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > maxArea) {
			maxArea = area;
			maxContour = contour;
		}
	}
	cv::Rect boundingBox = cv::boundingRect(maxContour);
	double scale = std::min(
		(double)image.cols / boundingBox.width,
		(double)image.rows / boundingBox.height);
	cv::Mat leafRegion = image(boundingBox);
	cv::Mat resizedLeaf;
	cv::resize(leafRegion, resizedLeaf, cv::Size(), scale, scale, cv::INTER_AREA);
	cv::Mat newImage(image.size(), image.type(), cv::Scalar::all(0));
	cv::Rect roi(
		(newImage.cols - resizedLeaf.cols) / 2,
		(newImage.rows - resizedLeaf.rows) / 2,
		resizedLeaf.cols,
		resizedLeaf.rows);
	resizedLeaf.copyTo(newImage(roi));
	image = newImage.clone();
}

cv::Mat ImageProcessing::CalculateGLCM(const cv::Mat& img)
{
	cv::Mat glcm = cv::Mat::zeros(256, 256, CV_32F);

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (y + 1 < 0 || y + 1 >= img.rows || x + 1 < 0 || x + 1 >= img.cols)
				continue;

			int rowValue = img.at<uchar>(y, x);
			int colValue = img.at<uchar>(y + 1, x + 1);
			if (rowValue < 256 && colValue < 256) {
				glcm.at<float>(rowValue, colValue) += 1.0f;
			}
		}
	}

	glcm = glcm / cv::sum(glcm)[0];
	return glcm;
}

std::vector<double> ImageProcessing::ExtractGLCMFeatures(const cv::Mat& glcm)
{
	double contrast = 0.0, dissimilarity = 0.0, homogeneity = 0.0;
	double asmFeature = 0.0, entropy = 0.0, correlation = 0.0;
	double idm = 0.0, clusterShade = 0.0, clusterProminence = 0.0;
	double maxProbability = 0.0, variance = 0.0;
	double sumAverage = 0.0, sumVariance = 0.0, sumEntropy = 0.0;
	double diffVariance = 0.0, diffEntropy = 0.0;
	double mean_i = 0.0, mean_j = 0.0, std_i = 0.0, std_j = 0.0;
	double N = static_cast<double>(glcm.rows);

	// Precompute some sums
	double sum_ij = 0.0, sum_p = 0.0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double p = glcm.at<float>(i, j);
			sum_ij += p * (i + j);
			sum_p += p;
			if (p > 0) entropy -= p * log(p);
			maxProbability = std::max(maxProbability, p);
		}
	}

	// Mean and standard deviation
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double p = glcm.at<float>(i, j);
			mean_i += i * p;
			mean_j += j * p;
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double p = glcm.at<float>(i, j);
			std_i += p * (i - mean_i) * (i - mean_i);
			std_j += p * (j - mean_j) * (j - mean_j);
			variance += (i - mean_i) * (i - mean_i) * p;
			if (i + j > 0) sumEntropy += (i + j) * p * log(i + j);
			if (i != j) diffEntropy += abs(i - j) * p * log(abs(i - j));
		}
	}
	std_i = sqrt(std_i);
	std_j = sqrt(std_j);

	// GLCM features
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double p = glcm.at<float>(i, j);
			double iMinusJ = abs(i - j);
			double iPlusJ = i + j;

			contrast += p * iMinusJ * iMinusJ;
			dissimilarity += p * iMinusJ;
			homogeneity += p / (1.0 + iMinusJ);
			asmFeature += p * p;
			idm += p / (1.0 + iMinusJ * iMinusJ);
			clusterShade += pow(iPlusJ - sum_ij, 3) * p;
			clusterProminence += pow(iPlusJ - sum_ij, 4) * p;
			sumAverage += iPlusJ * p;
			sumVariance += pow(iPlusJ - sumEntropy, 2) * p;
			if (i != j) diffVariance += iMinusJ * p;

			if (std_i != 0.0 && std_j != 0.0) {
				correlation += (i * j * p - mean_i * mean_j) / (std_i * std_j);
			}
		}
	}

	return {
		contrast, dissimilarity, homogeneity, asmFeature, entropy, correlation,
		idm, clusterShade, clusterProminence, maxProbability, variance,
		sumAverage, sumVariance, sumEntropy, diffVariance, diffEntropy
	};

	// Contrast: Measures the difference in brightness between a pixel and its neighbors across the entire image.
	// (A high value indicates a significant difference in brightness, suggesting more pronounced textures.)

	// Dissimilarity: Similar to contrast, but it gives more weight to differences in grayscale levels.

	// Homogeneity: Indicates how closely the elements of the GLCM are located to the diagonal of the matrix.
	// (High values mean the image is homogeneous.)
	
	// ASM (Angular Second Moment) or asmFeature: Measures the regularity or uniformity of grayscale levels.
	// (High values suggest greater uniformity.)

	// Entropy: Represents the disorder or complexity of the image.
	// (High entropy means more complexity in the texture of the image.)

	// Correlation: Measures how much a pixel is correlated with its neighbors across the entire image.
	// (High values indicate strong correlation.)

	// IDM (Inverse Difference Moment) or localized homogeneity: Measures the localization of homogeneity in the image.

	// Cluster Shade: An indicator of GLCM's asymmetry, it can be used to detect asymmetric textures in the image.

	// Cluster Prominence: Measures the asymmetry and prominence of GLCM elements.
	// (High values can indicate pronounced textures.)

	// Max Probability: The highest probability among the elements of GLCM, often used to measure uniformity.

	// Variance: Measures the variability of grayscale levels relative to the mean.

	// Sum Average: The average of the sums of grayscale levels.

	// Sum Variance: The variance of the sums of grayscale levels.

	// Sum Entropy: Measures the disorder or complexity of the sums of grayscale levels.

	// Difference Variance: Variance of the difference in grayscale levels.

	// Difference Entropy: Entropy of the difference in grayscale levels.
}

std::vector<double> ImageProcessing::ExtractTextureCaracteristics(const cv::Mat& image)
{
	std::vector<double> features;

	// Convert to grayscale
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	// Texture Features (GLCM - Gray-Level Co-occurrence Matrix)
	cv::Mat grayImageNormalized;
	cv::normalize(grayImage, grayImageNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
	// Calculate GLCM
	cv::Mat glcm = ImageProcessing::CalculateGLCM(grayImageNormalized);
	// Extract GLCM features
	std::vector<double> glcmFeatures = ImageProcessing::ExtractGLCMFeatures(glcm);
	features.insert(features.end(), glcmFeatures.begin(), glcmFeatures.end());

	return features;
}

std::vector<double> ImageProcessing::ExtractColorCaracteristics(const cv::Mat& image)
{
	std::vector<double> features;

	// Compute means and standard deviations for BGR channels
	cv::Scalar BGRMeans, BGRStdDevs;
	cv::meanStdDev(image, BGRMeans, BGRStdDevs);
	// Prepare the result vector
	std::vector<double> BGR = {
		BGRMeans[0], BGRMeans[1], BGRMeans[2],
		BGRStdDevs[0], BGRStdDevs[1], BGRStdDevs[2],
	};
	// Compute for each channel : min and max for BGR channels
	std::vector<cv::Mat> channelsBGR(3);
	cv::split(image, channelsBGR);
	for (int i = 0; i < 3; ++i) {
		double minValues, maxValues;
		cv::minMaxLoc(channelsBGR[i], &minValues, &maxValues);
		BGR.push_back(minValues);
		BGR.push_back(maxValues);
	}
	// Append the results to the features vector
	features.insert(features.end(), BGR.begin(), BGR.end());

	// Convert to HSV format
	cv::Mat HSVImage;
	cv::cvtColor(image, HSVImage, cv::COLOR_BGR2HSV);
	// Compute means and standard deviations for HSV channels
	cv::Scalar HSVMeans, HSVStdDevs;
	cv::meanStdDev(HSVImage, HSVMeans, HSVStdDevs);
	// Prepare the result vector
	std::vector<double> HSV = {
		HSVMeans[0], HSVMeans[1], HSVMeans[2],
		HSVStdDevs[0], HSVStdDevs[1], HSVStdDevs[2],
	};
	// Compute for each channel : min and max for HSV channels
	std::vector<cv::Mat> channelsHSV(3);
	cv::split(HSVImage, channelsHSV);
	for (int i = 0; i < 3; ++i) {
		double minValues, maxValues;
		cv::minMaxLoc(channelsHSV[i], &minValues, &maxValues);
		HSV.push_back(minValues);
		HSV.push_back(maxValues);
	}
	// Append the results to the features vector
	features.insert(features.end(), HSV.begin(), HSV.end());

	return features;
}