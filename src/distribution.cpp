#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sstream>

std::string checkImagesInDirectory(const std::string &directoryPath, std::string informations = "", size_t deepness = 0)
{
    size_t imageCount = 0;
    for (const auto &entry : std::filesystem::directory_iterator(directoryPath))
    {
        std::filesystem::path entryPath = entry.path();
        std::string DirectoryName = entryPath.filename().generic_string();

        if (std::filesystem::is_directory(entryPath))
        {
            for (size_t i = 0; i <= deepness; i++)
            {
                if (i == deepness)
                {
                    std::cout << "|--- ";
                }
                else
                {
                    std::cout << "|    ";
                }
            }

            std::string genericPath = entryPath.generic_string();
            std::string directoryName = entryPath.filename().generic_string();
            std::cout << directoryName << std::endl;

            informations += directoryName + ",";
            informations = checkImagesInDirectory(genericPath, informations, deepness + 1);
            informations += ",";
        }
        else if (std::filesystem::is_regular_file(entryPath))
        {
            std::string fileName = entryPath.filename().generic_string();

            size_t pos = fileName.find('.');
            if (pos != std::string::npos)
            {
                std::string fileType = fileName.substr(pos, fileName.size());
                if (fileType == ".JPG")
                {
                    imageCount++;
                }
            }
        }
    }
    for (size_t i = 0; i <= deepness; i++)
    {
        if (i == deepness)
        {
            std::cout << "'--- ";
        }
        else
        {
            std::cout << "|    ";
        }
    }

    std::cout << imageCount << " files" << std::endl;

    informations += std::to_string(imageCount);

    return informations;
}

// Fonction pour extraire les paires clé-valeur d'une chaîne
std::vector<std::pair<std::string, std::string>> extractKeyValuePairsFromString(const std::string &str)
{
    std::vector<std::pair<std::string, std::string>> result;
    std::istringstream ss(str);

    std::string segment;
    while (std::getline(ss, segment, ','))
    {
        try
        {
            size_t value = std::stoi(segment);
            for (auto it = result.end() - 1; it >= result.begin(); --it)
            {
                if (it->second == "")
                {
                    it->second = segment;
                    break;
                }
            }
        }
        catch (const std::invalid_argument &)
        {
            result.emplace_back(segment, "");
        }
        catch (const std::out_of_range &)
        {
            result.emplace_back(segment, "");
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    std::string directoryPath = argv[1];
    std::cout << directoryPath << std::endl;
    std::string output_str = directoryPath + "," + checkImagesInDirectory(directoryPath);
    std::cout << std::endl;

    std::vector<std::pair<std::string, std::string>> output_pair = extractKeyValuePairsFromString(output_str);

    for (const auto &pair : output_pair)
    {
        std::cout << "[" << pair.first << "," << pair.second << "]" << std::endl;
    }
    output_pair.erase(output_pair.begin());

    std::vector<std::string> colors = {"blue", "green", "red", "cyan", "magenta", "yellow", "purple", "orange", "pink", "brown", "gray"};

    // Construire la commande Python dans une chaîne de caractères
    std::string pythonCommand = "python -c \"import matplotlib.pyplot as plt; import ast; data = ast.literal_eval('[";

    for (const auto &pair : output_pair)
    {
        pythonCommand += "(\\\"" + pair.first + "\\\", " + pair.second + ",),";
    }
    // Supprimer la virgule finale
    pythonCommand.pop_back();

    // Ajouter le reste de la commande Python avec des couleurs aléatoires
    pythonCommand += "]'); keys, values = zip(*data); plt.figure(figsize=(14, 7)); plt.bar(keys, values, color=[";

    for (size_t i = 0; i < output_pair.size(); ++i)
    {
        pythonCommand += "\\\"" + colors[i % colors.size()] + "\\\", ";
    }

    // Supprimer la virgule finale
    pythonCommand.pop_back();
    pythonCommand.pop_back(); // Supprimer la virgule supplémentaire

    pythonCommand += "], width=1.0, align='center'); plt.xlabel('Keys'); plt.ylabel('Values'); plt.title('Bar Chart'); ";

    pythonCommand += "plt.figure(figsize=(7, 7)); plt.pie(values, labels = keys, colors = [";

    for (size_t i = 0; i < output_pair.size(); ++i)
    {
        pythonCommand += "\\\"" + colors[i % colors.size()] + "\\\", ";
    }

    // Supprimer la virgule finale
    pythonCommand.pop_back();
    pythonCommand.pop_back(); // Supprimer la virgule supplémentaire

    pythonCommand += "], autopct = '%1.1f%%', startangle = 140); plt.title('Pie Chart'); ";

    pythonCommand += "plt.show()\"  &";

    // Exécuter la commande Python depuis C++
    if (std::system(pythonCommand.c_str()) != 0)
    {
        std::cout << pythonCommand << std::endl
                  << std::endl;
    }

    return 0;
}
