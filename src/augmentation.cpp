#include <iostream>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

void applyRotate(cv::Mat &image)
{

    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(10, 70);

    double angle = distr(gen);

    // Calculer la taille de l'image r�sultante apr�s la rotation
    cv::Rect boundingRect = cv::RotatedRect(cv::Point2f(image.cols / 2.0, image.rows / 2.0), image.size(), angle).boundingRect();

    // Calculer automatiquement le facteur de scaling
    double scale_factor = std::min(static_cast<double>(image.cols) / boundingRect.width, static_cast<double>(image.rows) / boundingRect.height);

    // Calculer la matrice de rotation avec le scaling
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(image.cols / 2.0, image.rows / 2.0), angle, scale_factor);

    std::cout << "Original Image Size: " << image.size() << std::endl;
    std::cout << "Rotation Matrix:\n"
              << rotationMatrix << std::endl;
    std::cout << "Scale Factor: " << scale_factor << std::endl;

    // Cr�er une nouvelle image avec fond blanc de la taille de l'image d'origine
    cv::Mat rotatedImage = cv::Mat::zeros(image.size(), image.type());

    // Appliquer la rotation (et le scaling) � l'image d'origine
    cv::warpAffine(image, rotatedImage, rotationMatrix, rotatedImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    std::cout << "Rotated and Scaled Image Size: " << rotatedImage.size() << std::endl;

    // Remplacer l'image d'origine par l'image rot�e et r�duite
    image = rotatedImage;
}

void applyBlur(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(25, 35);

    double sigma = distr(gen) / 10.0;

    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void applyContrast(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(13, 20);

    double alpha = distr(gen) / 10.0;

    image.convertTo(image, -1, alpha, 0);
}

void applyScale(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(13, 17);

    double factor = distr(gen) / 10.0;

    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, factor);

    std::cout << "Original Image Size: " << image.size() << std::endl;
    std::cout << "Zoom Matrix:\n"
              << zoomMatrix << std::endl;
    std::cout << "Zoom Factor: " << factor << std::endl;

    // Appliquer le zoom � l'image d'origine
    cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    std::cout << "Zoomed Image Size: " << image.size() << std::endl;
}

void applyIllumination(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(20, 70);

    int brightness = distr(gen);

    image += cv::Scalar(brightness, brightness, brightness);
}

void applyProjective(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(10, 40);

    int topLY = distr(gen);
    int topRX = distr(gen);
    int botLX = distr(gen);
    int botRY = distr(gen);
    int botRX = distr(gen);

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

void putTextCentered(cv::Mat &img, const std::string &text, int y)
{
    int fontFace = cv::FONT_HERSHEY_COMPLEX;
    double fontScale = 0.8;
    int thickness = 1;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrigin((img.cols - textSize.width) / 2, y);

    cv::putText(img, text, textOrigin, fontFace, fontScale, CV_RGB(255, 255, 255), thickness);
}

int main(int argc, char *argv[])
{
    // Get the directory path from the command-line argument
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }
    std::string filePath = argv[1];

#ifdef _MSC_VER
    filePath = "images/Apple_Black_rot/image (56).JPG";
#endif

    // Load an image from the specified file path
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Unable to load the image." << std::endl;
        return -1;
    }

    // Create a vector to store multiple copies of the loaded image
    std::vector<cv::Mat> images;
    for (int i = 0; i < 7; i++)
    {
        images.push_back(image.clone());
    }

    // Apply various image processing operations to different copies of the image
    ImageProcessing::Rotate(images[1], 5.0, 45.0);
    ImageProcessing::Blur(images[2], 2.5, 3.5);
    ImageProcessing::Contrast(images[3], 1.25, 1.75);
    ImageProcessing::Scale(images[4], 1.25, 1.75);
    ImageProcessing::Illumination(images[5], 25, 35);
    ImageProcessing::Projective(images[6], 15, 30);
    applyRotate(applyRotateImage);
    applyBlur(applyBlurImage);
    applyContrast(applyContrastImage);
    applyScale(applyScaleImage);
    applyIllumination(applyIlluminationImage);

    // Create a mosaic image from the processed images
    ImageUtils::CreateImageMosaic(images, "Augmentations");

    // Save the processed images with their respective augmentation names
    images.erase(images.begin());
    std::vector<std::string> augmentations = {"Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective"};
    ImageUtils::SaveImages(filePath, images, augmentations);

    // Wait for a key press indefinitely (useful for viewing the result)
    applyProjective(applyProjectiveImage);

    std::vector<cv::Mat> images;
    images.push_back(image);
    images.push_back(applyRotateImage);
    images.push_back(applyBlurImage);
    images.push_back(applyContrastImage);
    images.push_back(applyScaleImage);
    images.push_back(applyIlluminationImage);
    images.push_back(applyProjectiveImage);

    int imageCount = 7;
    int titleOffset = 35;
    cv::Size imageSize = image.size();
    cv::Size displaySize(imageSize.width, imageSize.height + titleOffset);
    cv::Mat bigImage(displaySize.height, displaySize.width * imageCount, image.type(), cv::Scalar::all(0));

    std::string titles[] = {
        "Original", "Rotation", "Blur",
        "Contrast", "Scaling", "Illumination", "Projective"};

    for (int i = 0; i < imageCount; ++i)
    {
        cv::Rect roi(i * displaySize.width, titleOffset, imageSize.width, imageSize.height);
        cv::Mat targetROI = bigImage(roi);

        images[i].copyTo(targetROI);

        cv::Rect titleRegion(roi.x, 0, displaySize.width, titleOffset);
        cv::Mat titleROI = bigImage(titleRegion);
        putTextCentered(titleROI, titles[i], titleOffset - 10);
    }

    // Show the big image with titles
    cv::imshow("All Images", bigImage);
    cv::waitKey(0);

    return 0;
}
