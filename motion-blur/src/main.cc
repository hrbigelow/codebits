#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "blur.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <input_file> <viewportWidth> <viewportHeight> <output_file>\n", argv[0]);
    return 1;
  }
  char *path = argv[1];
  int viewportWidth = atoi(argv[2]);
  int viewportHeight = atoi(argv[3]);
  char *outPath = argv[4];

  int w, h, c;
  uchar3 *h_image = (uchar3 *)stbi_load(path, &w, &h, &c, 0);
  printf("image: %d x %d x %d\n", w, h, c);
  size_t outputSize = sizeof(uchar3) * viewportWidth * viewportHeight;
  uchar3 *h_blurred = (uchar3 *)malloc(sizeof(uchar3) * outputSize);

  // 3K shared mem per block.  Set this based on shared memory size and maximum
  // threads per SM
  uint numPixelLayers = 8;

  // how many timesteps in the trajectory you want to take for each occupied block
  // in the receptive field.
  float stepsPerOccuBlock = 3.0;  

  // trajectory
  uint numMats = 32;
  Homography trajectory[32];
  for (uint i=0; i != numMats; i++) {
    for (uint j=0; j!=3; j++) {
      for (uint k=0; k!=3; k++) {
        trajectory[i][j][k] = 0.0;
      }
    }
    trajectory[i][0][0] = 1.0 + 0.1 * i;
    trajectory[i][1][1] = 1.0 - 0.1 * i;
    trajectory[i][2][2] = 1.0;
  }
  motionBlur(trajectory, numMats, h_image, w, h, numPixelLayers, stepsPerOccuBlock, h_blurred,
      viewportWidth, viewportHeight);

  int channels = 3;
  stbi_write_png(outPath, viewportWidth, viewportHeight, channels, h_blurred, outputSize);

  free(h_blurred);
  free(h_image);

  return 0;
}

