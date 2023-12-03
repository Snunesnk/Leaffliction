#include "image_utils.h"

// Fonction pour créer une mosaïque d'images
void ImageUtils::CreateImageMosaic(std::vector<cv::Mat> images, std::string name) {
	int numCols = 7;
	int numRows = 1;

	// Créez des vecteurs de lignes et de colonnes pour empiler les images
	std::vector<cv::Mat> rows;
	for (int i = 0; i < numRows; i++) {
		std::vector<cv::Mat> cols;
		for (int j = 0; j < numCols; j++) {
			int index = i * numCols + j;
			if (index < images.size()) {
				cols.push_back(images[index]);
			}
		}
		// Empilez les images horizontalement dans une ligne
		cv::Mat row;
		cv::hconcat(cols, row);
		rows.push_back(row);
	}

	// Empilez les lignes verticalement pour créer la mosaïque
	cv::Mat mosaic;
	cv::vconcat(rows, mosaic);

	cv::imshow(name, mosaic);
}

void ImageUtils::SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types) {
	// Trouvez la position de la dernière barre oblique dans le chemin
	size_t lastSlashPosition = filePath.find_last_of('/');
	size_t lastPointPosition = filePath.find_last_of('.');

	// Extraire le répertoire en supprimant tout ce qui se trouve après la dernière barre oblique
	std::string saveDirectory = filePath.substr(0, lastSlashPosition + 1);

	// Extraire le nom du fichier sans l'extension
	std::string imageName = filePath.substr(lastSlashPosition + 1, lastPointPosition - lastSlashPosition - 1);

	for (int i = 0; i < images.size(); i++) {
		std::string filename = saveDirectory + imageName + "_" + types[i] + ".JPG";
		cv::imwrite(filename, images[i]);
		std::cout << "Saved : " << filename << std::endl;
	}
}