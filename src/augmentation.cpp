#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

void applyRotate(cv::Mat &image, double angle)
{

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

void applyBlur(cv::Mat &image, double sigma)
{
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void applyContrast(cv::Mat &image, double alpha)
{
    image.convertTo(image, -1, alpha, 0);
}

void applyScale(cv::Mat &image, double factor)
{
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

void applyIllumination(cv::Mat &image, int brightness)
{
    image += cv::Scalar(brightness, brightness, brightness);
}

void applyProjective(cv::Mat &image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 30);

    int topLY = distr(gen);
    int topLX = distr(gen);
    int topRY = distr(gen);
    int topRX = distr(gen);
    int botLY = distr(gen);
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
    dstPoints.push_back(cv::Point2f(topLY, topLX));
    dstPoints.push_back(cv::Point2f(image.cols - topRY, topRX));
    dstPoints.push_back(cv::Point2f(botLY, image.rows - botLX));
    dstPoints.push_back(cv::Point2f(image.cols - botRY, image.rows - botRX));

    // Get the Perspective Transform Matrix i.e. M
    cv::Mat warpMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    // Apply the perspective transformation to the image
    cv::warpPerspective(image, image, warpMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    filePath += "/Apple_Black_rot/image (55).JPG";

    // Charger une image � partir d'un fichier JPEG
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

    // V�rifier si l'image a �t� charg�e avec succ�s
    if (image.empty())
    {
        std::cerr << "Impossible de charger l'image." << std::endl;
        return -1;
    }

    cv::Mat applyRotateImage = image.clone();
    cv::Mat applyBlurImage = image.clone();
    cv::Mat applyContrastImage = image.clone();
    cv::Mat applyScaleImage = image.clone();
    cv::Mat applyIlluminationImage = image.clone();
    cv::Mat applyProjectiveImage = image.clone();

    applyRotate(applyRotateImage, 10.0);
    applyBlur(applyBlurImage, 3.0);
    applyContrast(applyContrastImage, 1.5);
    applyScale(applyScaleImage, 1.5);
    applyIllumination(applyIlluminationImage, 30);

    applyProjective(applyProjectiveImage);

    // Afficher l'image modifi�e
    cv::imshow("image", image);
    cv::imshow("applyRotateImage", applyRotateImage);
    cv::imshow("applyBlurImage", applyBlurImage);
    cv::imshow("applyContrastImage", applyContrastImage);
    cv::imshow("applyScaleImage", applyScaleImage);
    cv::imshow("applyIlluminationImage", applyIlluminationImage);
    cv::imshow("applyProjectiveImage", applyProjectiveImage);
    cv::waitKey(0);

    return 0;
}
