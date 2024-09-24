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

void matmul(const Homography &mat1, const Homography &mat2, Homography &result) {
  Homography tmp = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int i=0; i!=3; i++) {
    for (int j=0; j!=3; j++) {
      for (int k=0; k!=3; k++) {
        tmp[i][k] += mat1[i][j] * mat2[j][k];
      }
    }
  }
  for (int i=0; i!=3; i++){
    for (int j=0; j!=3; j++){
      result[i][j] = tmp[i][j];
    }
  }
}


void center_on_ref(Homography &mat, float ref_point_x, float ref_point_y) {
  // return a modified homography matrix "about the point" (ref_point_x, ref_point_y)
  Homography tx1 = { { 1, 0, -ref_point_x }, { 0, 1, -ref_point_y }, { 0, 0, 1 } };
  Homography tx2 = { { 1, 0, ref_point_x }, { 0, 1, ref_point_y }, { 0, 0, 1 } };
  matmul(mat, tx1, mat);
  matmul(tx2, mat, mat);
}


void print_mat(Homography &mat){
  for (int i=0; i!=3; i++) {
    printf("%5.2f %5.2f %5.2f\n", mat[i][0], mat[i][1], mat[i][2]);
  }
}
