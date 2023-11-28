#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

void applyRotate(cv::Mat& image, double angle) {

    // Calculer la taille de l'image résultante après la rotation
    cv::Rect boundingRect = cv::RotatedRect(cv::Point2f(image.cols / 2.0, image.rows / 2.0), image.size(), angle).boundingRect();

    // Calculer automatiquement le facteur de scaling
    double scale_factor = std::min(static_cast<double>(image.cols) / boundingRect.width, static_cast<double>(image.rows) / boundingRect.height);

    // Calculer la matrice de rotation avec le scaling
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(image.cols / 2.0, image.rows / 2.0), angle, scale_factor);

    std::cout << "Original Image Size: " << image.size() << std::endl;
    std::cout << "Rotation Matrix:\n" << rotationMatrix << std::endl;
    std::cout << "Scale Factor: " << scale_factor << std::endl;

    // Créer une nouvelle image avec fond blanc de la taille de l'image d'origine
    cv::Mat rotatedImage = cv::Mat::zeros(image.size(), image.type());

    // Appliquer la rotation (et le scaling) à l'image d'origine
    cv::warpAffine(image, rotatedImage, rotationMatrix, rotatedImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    std::cout << "Rotated and Scaled Image Size: " << rotatedImage.size() << std::endl;

    // Remplacer l'image d'origine par l'image rotée et réduite
    image = rotatedImage;
}

void applyBlur(cv::Mat& image, double sigma) {
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void applyContrast(cv::Mat& image, double alpha) {
    image.convertTo(image, -1, alpha, 0);
}

void applyZoom(cv::Mat& image, double zoom_factor) {
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, zoom_factor);

    std::cout << "Original Image Size: " << image.size() << std::endl;
    std::cout << "Zoom Matrix:\n" << zoomMatrix << std::endl;
    std::cout << "Zoom Factor: " << zoom_factor << std::endl;

    // Appliquer le zoom à l'image d'origine
    cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    std::cout << "Zoomed Image Size: " << image.size() << std::endl;
}

void applyIllumination(cv::Mat& image, int brightness) {
    image += cv::Scalar(brightness, brightness, brightness);
}

void applyProjective(cv::Mat& image, double alpha, double beta, double gamma, double dx, double dy, double dz, double f) {

    // Convertir les angles en radians
    alpha = (alpha - 90.0) * CV_PI / 180.0;
    beta = (beta - 90.0) * CV_PI / 180.0;
    gamma = (gamma - 90.0) * CV_PI / 180.0;

    // Obtenir la largeur et la hauteur pour faciliter l'utilisation dans les matrices
    double w = static_cast<double>(image.cols);
    double h = static_cast<double>(image.rows);

    // Projection 2D -> 3D matrix
    cv::Mat A1 = (cv::Mat_<double>(4, 3) <<
        1, 0, -w / 2,
        0, 1, -h / 2,
        0, 0, 0,
        0, 0, 1);

    // Rotation matrices around the X, Y, and Z axis
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, 0,
        0, cos(alpha), -sin(alpha), 0,
        0, sin(alpha), cos(alpha), 0,
        0, 0, 0, 1);

    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
        cos(beta), 0, -sin(beta), 0,
        0, 1, 0, 0,
        sin(beta), 0, cos(beta), 0,
        0, 0, 0, 1);

    cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
        cos(gamma), -sin(gamma), 0, 0,
        sin(gamma), cos(gamma), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Composed rotation matrix with (RX, RY, RZ)
    cv::Mat R = RX * RY * RZ;

    // Translation matrix
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1);

    // 3D -> 2D matrix
    cv::Mat A2 = (cv::Mat_<double>(3, 4) <<
        f, 0, w / 2, 0,
        0, f, h / 2, 0,
        0, 0, 1, 0);

    // Final transformation matrix
    cv::Mat trans = A2 * (T * (R * A1));

    // Apply matrix transformation
    cv::warpPerspective(image, image, trans, image.size(), cv::INTER_LANCZOS4);

}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    filePath += "/Apple_Black_rot/image (55).JPG";

    // Charger une image à partir d'un fichier JPEG
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

    // Vérifier si l'image a été chargée avec succès
    if (image.empty()) {
        std::cerr << "Impossible de charger l'image." << std::endl;
        return -1;
    }

    //applyRotate(image, 10.0);
    //applyBlur(image, 3.0);
    //applyContrast(image, 1.5);
    //applyScale(image, 1.5);
    //applyIllumination(image, 30);
    
    double alpha = 90; // Angle de rotation autour de l'axe X
    double beta = 45; // Angle de rotation autour de l'axe Y
    double gamma = 90; // Angle de rotation autour de l'axe Z
    double dx = 0; // Translation en x
    double dy = 0; // Translation en y
    double dz = 200; // Translation en z
    double f = 200; // Focale
    applyProjective(image, alpha, beta, gamma, dx, dy, dz, f);

    // Afficher l'image modifiée
    cv::imshow("Image modifiée", image);
    cv::waitKey(0);

    return 0;
}
