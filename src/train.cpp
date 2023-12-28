#include "image_utils.h"
#include "image_processing.h"
#include "model_utils.h"
#include "model_calculate.h"

#include <iostream>
#include <filesystem>
#include <chrono>

#include <zip_file.hpp>

std::vector<double> getCaracteristics(cv::Mat& image)
{
	std::vector<double> features;

	std::vector<double> color = ImageProcessing::ExtractColorCaracteristics(image);
	std::vector<double> texture = ImageProcessing::ExtractTextureCaracteristics(image);
	features.insert(features.end(), color.begin(), color.end());
	features.insert(features.end(), texture.begin(), texture.end());

	return features;
}

void GenerateDatabase(std::string source, std::vector<DataEntry>& database, int generation)
{
	std::cout << "\r\033[K" << "Database generation..." << std::endl;
	for (const auto& entry : std::filesystem::directory_iterator(source)) {
		// Get the target of the folder
		const std::string target = entry.path().filename().generic_string();
		// Next if not expected directory
		if (std::find(ModelUtils::targets.begin(), ModelUtils::targets.end(), target) == ModelUtils::targets.end()) {
			continue;
		}
		const std::string folderPath = entry.path().generic_string() + "/";
		const std::vector<std::string> imagePaths = ImageUtils::GetImagesInDirectory(folderPath, generation);

		std::vector<std::vector<std::string>> imageGroups;

		// Parallel processing to split	
		std::unordered_map<std::string, size_t> imageGroupIndices;
		cv::parallel_for_(cv::Range(0, imagePaths.size()), [&](const cv::Range& range) {
			for (int i = range.start; i < range.end; i++) {
				const std::string& imagePath = imagePaths[i];
				const size_t position = imagePath.find_last_of('_');
				if (position == std::string::npos || imagePath[position + 1] != 'T') {
					continue;
				}
				const std::string imageName = imagePath.substr(0, position);
				{
					std::lock_guard<std::mutex> lock(ImageUtils::mutex);
					if (imageGroupIndices.find(imageName) == imageGroupIndices.end()) {
						imageGroupIndices[imageName] = imageGroups.size();
						imageGroups.push_back({ imageName + ".JPG", imagePath });
					}
					else {
						imageGroups[imageGroupIndices[imageName]].push_back(imagePath);
					}
				}
			}
			});
		// Parallel processing to sort
		cv::parallel_for_(cv::Range(0, imageGroups.size()), [&](const cv::Range& range) {
			for (int i = range.start; i < range.end; i++) {
				std::vector<std::string>& imageGroup = imageGroups[i];
				std::sort(imageGroup.begin() + 1, imageGroup.end(), [](const std::string& a, const std::string& b) {
					std::string numeroA = a.substr(a.length() - 5, 1);
					std::string numeroB = b.substr(b.length() - 5, 1);
					int intA = std::stoi(numeroA);
					int intB = std::stoi(numeroB);
					return intA < intB;
					});
			}
			});

		// Generate database
		for (auto& imageGroup : imageGroups) {
			if (imageGroup.size() != 7) {
				throw std::runtime_error("Strange error");
			}
			DataEntry dataEntry;
			std::vector<std::vector<double>> featureGroups(imageGroup.size());
			// Parallel processing of each imageGroup
			cv::parallel_for_(cv::Range(0, imageGroup.size()), [&](const cv::Range& range) {
				for (int i = range.start; i < range.end; ++i) {
					// Get image
					std::string filePath = folderPath + imageGroup[i];
					cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
					if (image.empty()) {
						throw std::runtime_error("Unable to load the image: " + filePath);
					}
					// Add features 
					const std::vector<double> features = getCaracteristics(image);
					{
						std::lock_guard<std::mutex> lock(ImageUtils::mutex);
						for (const auto feature : features) {
							featureGroups[i].push_back(feature);
						}
						// Display name
						std::cout << "\r\033[K" << filePath << std::flush;
					}
				}
				});
			// Add dataEntry to the database
			dataEntry.index = database.size();
			dataEntry.target = target;
			for (const auto featureGroup : featureGroups) {
				for (const auto feature : featureGroup) {
					dataEntry.features.push_back(feature);
				}
			}
			database.push_back(dataEntry);
			// Update and display progression
			int progress = (database.size() * 100) / ((generation / 7) * 8);
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n" << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%";
			std::cout << "\033[A";
		}
	}
	std::cout << "\r\033[K" << "DataEntry generated : " << database.size() << std::endl;
}

void DeleteExistingImages(const std::vector<std::filesystem::directory_entry>& filesystemDirectories)
{
	std::cout << "Reseting..." << std::endl;
	ImageUtils::numComplete = 0;
	// Count files
	cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
			for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
				const std::string imageName = entry.path().filename().string();
				if (imageName.find('_') != std::string::npos && entry.is_regular_file() && entry.path().extension() == ".JPG") {
					{
						std::lock_guard<std::mutex> lock(ImageUtils::mutex);
						ImageUtils::numComplete++;
					}
				}
			}
		}
		});
	ImageUtils::progress = 0;
	// Delete files
	cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
			for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
				const std::string imageName = entry.path().filename().string();
				if (imageName.find('_') != std::string::npos && entry.is_regular_file() && entry.path().extension() == ".JPG") {
					std::filesystem::remove(entry.path());
					{
						// Progression
						std::lock_guard<std::mutex> lock(ImageUtils::mutex);
						std::cout << "\r\033[K" << "Delete file : " << entry.path().generic_string();
						int progress = (++ImageUtils::progress) * 100 / ImageUtils::numComplete;
						int numComplete = (progress * 50) / 100;
						int numRemaining = 50 - numComplete;
						std::cout << "\n" << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
						std::cout << "\033[A";
					}
				}
			}
		}
		});
	std::cout << std::endl << "\r\033[K" << "\033[A" << "\r\033[K" << "\033[A" << "\r\033[K" << "Images deleted : " << ImageUtils::progress << std::endl;
}

void GenerateAugmentations(const std::vector<std::filesystem::directory_entry>& filesystemDirectories, int generation)
{
	std::cout << "\r\033[K" << "Augmentations..." << std::endl;
	std::vector<double> augGenerations(ModelUtils::targets.size(), generation);
	// Count files
	cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
			for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
				if (entry.is_regular_file() && entry.path().extension() == ".JPG") {
					if (--augGenerations[directory] == 0) {
						break;
					}
				}
			}
		}
		});
	ImageUtils::numComplete = 0;
	for (auto& augGeneration : augGenerations) {
		ImageUtils::numComplete += augGeneration;
	}
	ImageUtils::progress = 0;
	// Augmentation files
	cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
			if (augGenerations[directory] > 0) {
				ImageUtils::SaveAFromToDirectory(directoryPath, directoryPath, augGenerations[directory]);
			}
		}
		});
	std::cout << "\r\033[K" << "\033[A" << "\r\033[K" << "Augmentations generated : " << ImageUtils::progress << std::endl;
}

void GenerateTransformations(const std::vector<std::filesystem::directory_entry>& filesystemDirectories, int generation)
{
	std::cout << "\r\033[K" << "Transformation..." << std::endl;
	ImageUtils::numComplete = generation * 8;
	ImageUtils::progress = 0;
	// Transformation files
	cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
			ImageUtils::SaveTFromToDirectory(directoryPath, directoryPath, generation);
		}
		});
	std::cout << "\r\033[K" << "\033[A" << "\r\033[K" << "Transformations generated : " << ImageUtils::progress * 6 << std::endl;
}

void GenerateZip(const std::vector<std::filesystem::directory_entry>& filesystemDirectories, const std::string& source, const std::string& models)
{
	try {
		ImageUtils::numComplete = 0;
		cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
			for (int directory = range.start; directory < range.end; directory++) {
				const std::string directoryPath = filesystemDirectories[directory].path().generic_string() + "/";
				for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
					const std::string imageName = entry.path().filename().string();
					if (imageName.find('_') != std::string::npos && entry.is_regular_file() && entry.path().extension() == ".JPG") {
						{
							std::lock_guard<std::mutex> lock(ImageUtils::mutex);
							ImageUtils::numComplete++;
						}
					}
				}
			}});
		ImageUtils::progress = 0;
		cv::parallel_for_(cv::Range(0, filesystemDirectories.size()), [&](const cv::Range& range) {
			for (int directory = range.start; directory < range.end; directory++) {
				miniz_cpp::zip_file zip;
				for (const auto& entry : std::filesystem::directory_iterator(filesystemDirectories[directory])) {
					const std::string imageName = entry.path().filename().string();
					if (imageName.find('_') != std::string::npos && entry.is_regular_file() && entry.path().extension() == ".JPG") {
						// calculate the relative path with respect to the base directory
						std::string relativePath = std::filesystem::relative(entry.path(), source).generic_string();
						zip.write(entry.path().string(), relativePath);
						{
							std::lock_guard<std::mutex> lock(ImageUtils::mutex);
							int progress = (++ImageUtils::progress) * 100 / ImageUtils::numComplete;
							int numComplete = (progress * 50) / 100;
							int numRemaining = 50 - numComplete;
							std::cout << "\r\033[K" << "Compressed : " << relativePath << std::flush;
							std::cout << "\n" << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
							std::cout << "\033[A";
						}
					}
				}
				zip.save(ModelUtils::targets[directory] + ".zip");
			}
			});
		std::cout << "\n" << "\r\033[K" << "\033[A" << "\r\033[K";

		miniz_cpp::zip_file zip;
		zip.writestr("models.txt", models);
		zip.save("models.zip");

		system("find . -maxdepth 2 -name '*.zip' -exec zip archive.zip '{}' \\; -exec rm '{}' \\;");
	}
	catch (const std::exception& e) {
		std::cerr << std::endl << e.what() << std::endl;
		std::cerr << "Trying by console command... " << std::endl;

		miniz_cpp::zip_file zip;
		zip.writestr("models.txt", models);
		zip.save("archive.zip");

		system("find . -maxdepth 2 -type f -name '*_*' -exec zip archive.zip {} +");
	}
}

int main(int argc, char* argv[])
{
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " <source_path> -gen <generation_max> -csv <csv_path>");
		}
		// Apple_Black_rot     620 files
		// Apple_healthy       1640 files
		// Apple_rust          275 files
		// Apple_scab          629 files
		// Grape_Black_rot     1178 files
		// Grape_Esca          1382 files
		// Grape_healthy       422 files
		// Grape_spot          1075 files
		std::string source = argv[1];
		std::string csv;// = "data.csv";
		int generation = 1640;
		bool reset = false;

		// Parse command-line arguments
		for (int i = 2; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-gen" && i + 1 < argc) {
				generation = std::atoi(argv[i + 1]);
				if (generation > 1640) {
					generation = 1640;
				}
				++i;
			}
			else if (arg == "-csv" && i + 1 < argc) {
				csv = arg[i + 1];
				++i;
			}
			else if (arg == "-h") {
				std::cout << "Usage: " << argv[0] << " -gen <generation_max> -csv <csv_path> --data" << std::endl;
				return 0;
			}
		}

		std::vector<DataEntry> database;

		// Get folder list
		std::vector<std::filesystem::directory_entry> filesystemDirectories;
		for (const auto entry : std::filesystem::directory_iterator(source)) {
			const std::string directoryName = entry.path().filename().generic_string();
			// Check only for expected directories
			if (std::filesystem::is_directory(entry.path()) && std::find(ModelUtils::targets.begin(), ModelUtils::targets.end(), directoryName) == ModelUtils::targets.end()) {
				continue;
			}
			filesystemDirectories.push_back(entry);
		}

		if (csv.empty()) {
			auto start_time = std::chrono::high_resolution_clock::now();

			std::cout << "Images generation..." << std::endl;
			if (reset == true) {
				DeleteExistingImages(filesystemDirectories);
				GenerateAugmentations(filesystemDirectories, generation);
				GenerateTransformations(filesystemDirectories, generation);
			}
			GenerateDatabase(source, database, generation * 7);
			ModelUtils::SaveDataFile("data.csv", database);

			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
			std::cout << "\r\033[K" << "Images generation took " << duration * 0.000001 << " seconds." << std::endl;
		}
		else {
			ModelUtils::LoadDataFile(database, csv);
		}

		auto start_time = std::chrono::high_resolution_clock::now();

		std::vector<std::vector<double>> weights(ModelUtils::targets.size(), std::vector<double>(database[0].features.size(), 0.0));
		std::vector<double> featureMeans, featureStdDevs;
		ModelCalculate::GenerateModels(database, weights, featureMeans, featureStdDevs);

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
		std::cout << "\r\033[K" << "Models training took " << duration * 0.000001 << " seconds." << std::endl;

		// ZIP
		std::cout << "ZIP generation..." << std::endl;
		std::string models = ModelUtils::SaveModels(weights, featureMeans, featureStdDevs);
		GenerateZip(filesystemDirectories, source, models);
		std::cout << "\r\033[K" << "\033[A" << "\r\033[K" << "ZIP generated." << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}