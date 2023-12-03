#include <iostream>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

/// Possibility to give an image path or a dir path
/// If image path, then apply augmentation to image and show it
/// If dir path, apply augmentation to all images but do not show it
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

    // Create a mosaic image from the processed images
    ImageUtils::CreateImageMosaic(images, "Augmentations");

    // Save the processed images with their respective augmentation names
    images.erase(images.begin());
    std::vector<std::string> augmentations = {"Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective"};
    ImageUtils::SaveImages(filePath, images, augmentations);

    return 0;
}
