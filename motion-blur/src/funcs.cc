#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "funcs.h"
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

void make_homography(
    float tx, float ty, float sx, float sy, float deg, float sk, float px, float py, 
    float ref_point_x, float ref_point_y, Homography &out) {
    // create a homography matrix with the desired translation, scale, rotation
    // (degrees), projection, about (ref_point_x, ref_point_y), storing in `out`
    float rad = deg * (2.0 * M_PI) / 360.0;
    float cos_theta = std::cos(rad);
    float sin_theta = std::sin(rad);
    Homography P = { { 1, 0, 0 }, { 0, 1, 0 }, { px, py, 1 } };
    Homography T = { { 1, 0, tx }, { 0, 1, ty }, { 0, 0, 1 } };
    Homography R = { { cos_theta, -sin_theta, 0 }, { sin_theta, cos_theta, 0 }, { 0, 0, 1 } };
    Homography K = { { 1, sk, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    Homography S = { { sx, 0, 0 }, { 0, sy, 0 }, { 0, 0, 1 } };

    matmul(P, T, out);
    matmul(R, out, out);
    matmul(K, out, out);
    matmul(S, out, out);
    center_on_ref(out, ref_point_x, ref_point_y);
}


void print_mat(Homography &mat){
  for (int i=0; i!=3; i++) {
    printf("%5.2f %5.2f %5.2f\n", mat[i][0], mat[i][1], mat[i][2]);
  }
}


std::pair<float, float> 
lerp(const std::pair<float, float> &p0, const std::pair<float, float> &p1, float t) {
    return {
        (1 - t) * p0.first + t * p1.first,
        (1 - t) * p0.second + t * p1.second
    };
}

float bezier(const BezierPoints &points, float t) {
    auto p0p1 = lerp(points[0], points[1], t);
    auto p1p2 = lerp(points[1], points[2], t);
    auto p2p3 = lerp(points[2], points[3], t);
    auto p01_12 = lerp(p0p1, p1p2, t);
    auto p12_23 = lerp(p1p2, p2p3, t);
    auto last = lerp(p01_12, p12_23, t);
    return last.second;
}


