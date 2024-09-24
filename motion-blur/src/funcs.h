#ifndef _FUNCS_H
#define _FUNCS_H

#include "dims.h" 

bool loadHomography(const char *filename, Homography **mats, unsigned int *numMatrices);
void center_on_ref(Homography &mat, float ref_point_x, float ref_point_y);
void print_mat(Homography &mat);

#endif // _FUNCS_H

