#ifndef _FUNCS_H
#define _FUNCS_H
#include <array>
#include <utility>

#include "dims.h" 

bool loadHomography(const char *filename, Homography **mats, unsigned int *numMatrices);
void center_on_ref(Homography &mat, float ref_point_x, float ref_point_y);
void print_mat(Homography &mat);
void make_homography(
    float tx, float ty, float sx, float sy, float deg, float k, float px, float py, 
    float ref_point_x, float ref_point_y, Homography &out);

typedef std::array<std::pair<float, float>, 4> BezierPoints;
float bezier(const BezierPoints &points, float t);

#endif // _FUNCS_H

