#include <iostream>

#include "vector.h"

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
		return 1;
	}
	std::size_t size = std::stoull(argv[1]);

	Vector v;
	for (std::size_t i = 0; i < size; ++i) {
		v.push_back(i);
	}

	std::cout << "Vector size: " << v.size() << std::endl;
	std::cin.get();

	return 0;
}
