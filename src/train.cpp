
#include <iostream>

int main(int argc, char* argv[]) {
	try {
		if (argc != 2) {
			std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
			return 1;
		}

	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}