#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

// Convertit l'image en niveaux de gris
void applyConvertToGrayScale(cv::Mat& image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

// Applique un filtre de couleur pour la segmentation
void applyColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound) {
    // Convertit l'image en espace de couleur HSV
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Crée un masque pour la plage de couleur spécifiée
    cv::Mat colorMask;
    cv::inRange(hsvImage, lowerBound, upperBound, colorMask);

    // Met en rouge les parties de l'image qui correspondent au masque
    image.setTo(cv::Scalar(0, 0, 255), colorMask);
}

// Égalise l'histogramme de l'image en niveaux de gris
void applyEqualizeHistogram(cv::Mat& image) {
    // Convertit l'image en niveaux de gris
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // Égalise l'histogramme
    cv::equalizeHist(image, image);
}

// Applique un seuillage adaptatif à l'image en niveaux de gris
void applyAdaptiveThresholding(cv::Mat& image, int blockSize, double c) {
    // Convertit l'image en niveaux de gris
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Applique le seuillage adaptatif
    cv::adaptiveThreshold(grayImage, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, c);
}

// Détecte les points clés avec des descripteurs ORB et les dessine sur l'image
void applyDetectORBKeyPoints(cv::Mat& image) {
    // Crée un détecteur ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Détecte les points clés
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(image, keypoints);

    // Dessine les points clés sur l'image
    cv::drawKeypoints(image, keypoints, image,  cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

void applyPreserveMaskedRegion(cv::Mat& image, cv::Scalar objectColorLower, cv::Scalar objectColorUpper) {
    // Convertit l'image en espace de couleur HSV
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Crée un masque pour la plage de couleur de l'objet
    cv::Mat objectMask;
    cv::inRange(hsvImage, objectColorLower, objectColorUpper, objectMask);

    // Crée une copie de l'image originale
    cv::Mat originalImage = image.clone();

    // Inverse le masque pour conserver la zone blanche
    cv::bitwise_not(objectMask, objectMask);

    // Met en blanc les parties de l'image qui correspondent au masque
    originalImage.setTo(cv::Scalar(255, 255, 255), objectMask);

    // Fusionne l'image transformée avec l'image originale
    cv::addWeighted(originalImage, 1.0, image, 0.0, 0.0, image);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    filePath += "/Apple_Black_rot/image (71).JPG";

    // Charger une image à partir d'un fichier JPEG
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

    // Vérifier si l'image a été chargée avec succès
    if (image.empty()) {
        std::cerr << "Impossible de charger l'image." << std::endl;
        return -1;
    }

    cv::Mat applyConvertToGrayScaleImage = image.clone();
    cv::Mat applyColorFilteringImage = image.clone();
    cv::Mat applyEqualizeHistogramImage = image.clone();
    cv::Mat applyAdaptiveThresholdingImage = image.clone();
    cv::Mat applyDetectORBKeyPointsImage = image.clone();
    cv::Mat applyPreserveMaskedRegionImage = image.clone();


    applyConvertToGrayScale(applyConvertToGrayScaleImage);
    applyColorFiltering(applyColorFilteringImage, cv::Scalar(30, 50, 50), cv::Scalar(90, 255, 255));
    applyEqualizeHistogram(applyEqualizeHistogramImage);
    applyAdaptiveThresholding(applyAdaptiveThresholdingImage, 51, 10);
    applyDetectORBKeyPoints(applyDetectORBKeyPointsImage);
    applyPreserveMaskedRegion(applyPreserveMaskedRegionImage, cv::Scalar(0, 50, 50), cv::Scalar(30, 255, 255));


    cv::imshow("image", image);
    cv::imshow("applyConvertToGrayScaleImage", applyConvertToGrayScaleImage);
    cv::imshow("applyColorFilteringImage", applyColorFilteringImage);
    cv::imshow("applyEqualizeHistogramImage", applyEqualizeHistogramImage);
    cv::imshow("applyAdaptiveThresholdingImage", applyAdaptiveThresholdingImage);
    cv::imshow("applyDetectORBKeyPointsImage", applyDetectORBKeyPointsImage);
    cv::imshow("applyPreserveMaskedRegionImage", applyPreserveMaskedRegionImage);

    cv::waitKey(0);
}