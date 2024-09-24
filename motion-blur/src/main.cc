#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "blur.h"
#include "funcs.h"

int main(int argc, char **argv) {
  if (argc != 9) {
    printf("Usage: %s <input_file> <trajectory_file> <viewport_width> <viewport_height> "
        "<steps_per_occu_block> <ref_point_x> <ref_point_y> <output_file>\n", argv[0]);
    return 1;
  }
  char *path = argv[1];
  char *trajectory_file = argv[2];
  int viewportWidth = atoi(argv[3]);
  int viewportHeight = atoi(argv[4]);

  // how many timesteps in the trajectory you want to take for each occupied block
  // in the receptive field.
  float stepsPerOccuBlock = atof(argv[5]);
  float ref_point_x = atof(argv[6]);
  float ref_point_y = atof(argv[7]);
  char *outPath = argv[8];

  int w, h, c;
  uchar3 *h_image = (uchar3 *)stbi_load(path, &w, &h, &c, 0);
  printf("image: %d x %d x %d\n", w, h, c);
  size_t outputSize = sizeof(uchar3) * viewportWidth * viewportHeight;
  uchar3 *h_blurred = (uchar3 *)malloc(sizeof(uchar3) * outputSize);

  // 3K shared mem per block.  Set this based on shared memory size and maximum
  // threads per SM
  uint numPixelLayers = 8;


  // trajectory
  Homography *trajectory;
  uint numMats;
  loadHomography(trajectory_file, &trajectory, &numMats);
  
  for (int m=0; m!= numMats; m++){
    center_on_ref(trajectory[m], ref_point_x, ref_point_y);
    print_mat(trajectory[m]);
    printf("\n");
  }


  motionBlur(trajectory, numMats, h_image, w, h, numPixelLayers, stepsPerOccuBlock, h_blurred,
      viewportWidth, viewportHeight);

  int channels = 3;
  unsigned stride_in_bytes = viewportWidth * 3;
  stbi_write_png(outPath, viewportWidth, viewportHeight, channels, h_blurred, stride_in_bytes);

  free(h_blurred);
  free(h_image);

  return 0;
}

