#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <argparse/argparse.hpp>

#include "blur.h"
#include "funcs.h"
#include <vector>

float interpolate(float beg, float end, float weight) {
    return beg + (end - beg) * weight;
}

int main(int argc, char *argv[]) {
    
    argparse::ArgumentParser program("motion_blur");
    program.add_argument("input_image")
        .help("source PNG image file");

    program.add_argument("output_image")
        .help("Output PNG filename");

    program.add_argument("-vp", "--viewport")
        .help("output viewport (width, height)")
        .nargs(2)
        .default_value(std::vector<unsigned int>{500, 250})
        .scan<'d', unsigned int>();

    program.add_argument("-sb", "--steps-per-block")
        .help("Trajectory sampling density in steps per occupied (32 x 32) pixel block")
        .default_value(10.0f)
        .scan<'g', float>();

    program.add_argument("-ref", "--reference-point")
        .help("Reference point (x, y) about which the homographies are taken")
        .nargs(2)
        .default_value(std::vector<float>{250, 125})
        .scan<'g', float>();

    program.add_argument("-tx", "--translate-x")
        .help("beginning and ending horizontal translation in pixels")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-ty", "--translate-y")
        .help("beginning and ending vertical translation in pixels")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-sx", "--scale-x")
        .help("beginning and ending horizontal scale factor")
        .nargs(2)
        .default_value(std::vector<float>{1, 1})
        .scan<'g', float>();

    program.add_argument("-sy", "--scale-y")
        .help("beginning and ending vertical scale factor")
        .nargs(2)
        .default_value(std::vector<float>{1, 1})
        .scan<'g', float>();

    program.add_argument("-r", "--rotate")
        .help("beginning and ending rotation angle in degrees")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-sk", "--skew")
        .help("beginning and ending skew")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-px", "--project-x")
        .help("beginning and ending horizontal projection")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-py", "--project-y")
        .help("beginning and ending vertical projection")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-pbuf", "--pixel-buf-bytes")
        .help("size in bytes for shared memory pixel buffer")
        .default_value((int)(1024 * 44))
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto input_file = program.get<std::string>("input_image");
    auto output_file = program.get<std::string>("output_image");
    auto steps_per_occu_block = program.get<float>("sb");
    auto viewport = program.get<std::vector<unsigned int>>("-vp");
    auto ref_point = program.get<std::vector<float>>("-ref");
    auto translate_x = program.get<std::vector<float>>("-tx");
    auto translate_y = program.get<std::vector<float>>("-ty");
    auto scale_x = program.get<std::vector<float>>("-sx");
    auto scale_y = program.get<std::vector<float>>("-sy");
    auto rotate = program.get<std::vector<float>>("-r");
    auto skew = program.get<std::vector<float>>("-sk");
    auto project_x = program.get<std::vector<float>>("-px");
    auto project_y = program.get<std::vector<float>>("-py");
    auto pixel_buf_bytes = program.get<int>("-pbuf");

    int w, h, c;
    uchar3 *h_image = (uchar3 *)stbi_load(input_file.c_str(), &w, &h, &c, 0);
    printf("image: %d x %d x %d\n", w, h, c);
    size_t outputSize = sizeof(uchar3) * viewport[0] * viewport[1];
    uchar3 *h_blurred = (uchar3 *)malloc(sizeof(uchar3) * outputSize);

    Homography trajectory[MAX_HOMOGRAPHY_MATS];
    float inc = 1.0 / (float)(MAX_HOMOGRAPHY_MATS - 1);
    float t;
    for (int i = 0; i != MAX_HOMOGRAPHY_MATS; i++) {
        t = i * inc;
        make_homography(
                interpolate(translate_x[0], translate_x[1], t),
                interpolate(translate_y[0], translate_y[1], t),
                interpolate(scale_x[0], scale_x[1], t),
                interpolate(scale_y[0], scale_y[1], t),
                interpolate(rotate[0], rotate[1], t),
                interpolate(skew[0], skew[1], t),
                interpolate(project_x[0], project_x[1], t),
                interpolate(project_y[0], project_y[1], t),
                ref_point[0], ref_point[1],
                trajectory[i]);
    }

    motionBlur(trajectory, MAX_HOMOGRAPHY_MATS, h_image, w, h, pixel_buf_bytes,
            steps_per_occu_block, h_blurred, viewport[0], viewport[1]);

    int channels = 3;
    unsigned stride_in_bytes = viewport[0] * 3;
    stbi_write_png(output_file.c_str(), viewport[0], viewport[1], channels,
            h_blurred, stride_in_bytes);

    free(h_blurred);
    free(h_image);

    return 0;
}

