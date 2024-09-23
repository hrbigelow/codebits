#include <iostream>
#include <fstream>
#include <sstream>
#include "blur.h"

// Function to load matrices from the file
bool loadHomography(const char *filename, Homography **matrices, unsigned int *numMatrices) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    file >> *numMatrices;
    if (*numMatrices <= 0) {
        std::cerr << "Invalid number of matrices: " << *numMatrices << std::endl;
        return false;
    }

    *matrices = (Homography *)malloc(sizeof(Homography) * *numMatrices);

    // Parse the matrix data
    for (int i = 0; i < *numMatrices; ++i) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                if (!(file >> (*matrices)[i][row][col])) {
                    std::cerr << "Error reading matrix data." << std::endl;
                    return false;
                }
            }
        }
    }

    file.close();
    return true;
}
