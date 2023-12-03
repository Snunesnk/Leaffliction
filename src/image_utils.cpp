#include "image_utils.h"

// Fonction pour cr�er une mosa�que d'images
void ImageUtils::CreateImageMosaic(std::vector<cv::Mat> images, std::string name)
{
	int imageCount = 7;
	int titleOffset = 35;
	cv::Size imageSize = images[0].size();
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
	cv::imshow(name, bigImage);
}

void ImageUtils::SaveImages(const std::string &filePath, const std::vector<cv::Mat> &images, const std::vector<std::string> &types)
{
	// Trouvez la position de la derni�re barre oblique dans le chemin
	size_t lastSlashPosition = filePath.find_last_of('/');
	size_t lastPointPosition = filePath.find_last_of('.');

	// Extraire le r�pertoire en supprimant tout ce qui se trouve apr�s la derni�re barre oblique
	std::string saveDirectory = filePath.substr(0, lastSlashPosition + 1);

	// Extraire le nom du fichier sans l'extension
	std::string imageName = filePath.substr(lastSlashPosition + 1, lastPointPosition - lastSlashPosition - 1);

	for (int i = 0; i < images.size(); i++)
	{
		std::string filename = saveDirectory + imageName + "_" + types[i] + ".JPG";
		cv::imwrite(filename, images[i]);
		std::cout << "Saved : " << filename << std::endl;
	}
}