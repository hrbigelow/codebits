#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <argparse/argparse.hpp>

#include "blur.h"
#include "funcs.h"
#include <vector>
#include <sstream>
#include <functional>

float interpolate(float beg, float end, float weight) {
    return beg + (end - beg) * weight;
}

int main(int argc, char *argv[]) {
    std::ostringstream out_opts, output_path;
    
    argparse::ArgumentParser program("motion_blur");
    program.add_argument("input_image")
        .help("source PNG image file");

    program.add_argument("output_dir")
        .help("Output PNG directory");

    program.add_argument("-vp", "--viewport")
        .help("output viewport (width, height) (default (0, 0) will be set to input image size)")
        .nargs(2)
        .default_value(std::vector<unsigned int>{0, 0})
        .scan<'d', unsigned int>();

    program.add_argument("-sb", "--steps-per-block")
        .help("Trajectory sampling density in steps per occupied (32 x 32) pixel block")
        .default_value(10.0f)
        .scan<'g', float>();

    program.add_argument("-ref", "--reference-point")
        .help("Reference point (x, y) about which the homographies are taken")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-tx", "--translate-x")
        .help("start and end horizontal translation in pixels")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-ty", "--translate-y")
        .help("start and end vertical translation in pixels")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-sx", "--scale-x")
        .help("start and end horizontal scale factor")
        .nargs(2)
        .default_value(std::vector<float>{1, 1})
        .scan<'g', float>();

    program.add_argument("-sy", "--scale-y")
        .help("start and end vertical scale factor")
        .nargs(2)
        .default_value(std::vector<float>{1, 1})
        .scan<'g', float>();

    program.add_argument("-r", "--rotate")
        .help("start and end rotation angle in degrees")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-sk", "--skew")
        .help("start and end skew")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-px", "--project-x")
        .help("start and end horizontal projection")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-py", "--project-y")
        .help("start and end vertical projection")
        .nargs(2)
        .default_value(std::vector<float>{0, 0})
        .scan<'g', float>();

    program.add_argument("-pbuf", "--pixel-buf-bytes")
        .help("size in bytes for shared memory pixel buffer")
        .default_value((int)(1024 * 44))
        .scan<'i', int>();

    program.add_argument("-em", "--exposure-multiplier")
        .help("factor to multiply color intensity before saturation cap")
        .default_value(1.0f)
        .scan<'g', float>();

    program.add_argument("-wig", "--wiggle")
        .help("If present, produce a wiggled path for translation")
        .implicit_value(true)
        .default_value(false);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto input_file = program.get<std::string>("input_image");
    auto output_dir = program.get<std::string>("output_dir");
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
    auto wiggle = program.get<bool>("-wig");
    auto exposure_mul = program.get<float>("-em");
    auto pixel_buf_bytes = program.get<int>("-pbuf");

    output_path << output_dir << "/";
    auto last_dot = input_file.find_last_of('.');
    out_opts << input_file.substr(0, last_dot);
    if (steps_per_occu_block != 10.0)
        out_opts << " -sb " << steps_per_occu_block;
    if (ref_point[0] != 0 || ref_point[1] != 0)
        out_opts << " -ref " << ref_point[0] << " " << ref_point[1]; 
    if (translate_x[0] != 0.0 || translate_x[1] != 0.0) 
        out_opts << " -tx " << translate_x[0] << " " << translate_x[1];
    if (translate_y[0] != 0.0 || translate_y[1] != 0.0) 
        out_opts << " -ty " << translate_y[0] << " " << translate_y[1];
    if (scale_x[0] != 1.0 || scale_x[1] != 1.0)
        out_opts << " -sx " << scale_x[0] << " " << scale_x[1];
    if (scale_y[0] != 1.0 || scale_y[1] != 1.0)
        out_opts << " -sy " << scale_y[0] << " " << scale_y[1];
    if (rotate[0] != 0.0 || rotate[1] != 0.0)
        out_opts << " -r " << rotate[0] << " " << rotate[1];
    if (skew[0] != 0.0 || skew[1] != 0.0) 
        out_opts << " -sk " << skew[0] << " " << skew[1];
    if (project_x[0] != 0.0 || project_x[1] != 0.0)
        out_opts << " -px " << project_x[0] << " " << project_x[1];
    if (project_y[0] != 0.0 || project_y[1] != 0.0)
        out_opts << " -py " << project_y[0] << " " << project_y[1];
    if (exposure_mul != 1.0)
        out_opts << " -em " << exposure_mul;
    if (wiggle)
        out_opts << " -wig";

    auto name = alpha_numeric(std::hash<std::string>{}(out_opts.str()));
    output_path << name << ".png";

    int w, h, c;
    uchar3 *h_image = (uchar3 *)stbi_load(input_file.c_str(), &w, &h, &c, 0);
    std::cerr << "image: " << w << " x " << h << " x " << c << std::endl;
    if (viewport[0] == 0 && viewport[1] == 0) {
        viewport[0] = w;
        viewport[1] = h;
    }
    size_t outputSize = sizeof(uchar3) * viewport[0] * viewport[1];
    uchar3 *h_blurred = (uchar3 *)malloc(sizeof(uchar3) * outputSize);

    Homography trajectory[MAX_HOMOGRAPHY_MATS];
    float inc = 1.0 / (float)(MAX_HOMOGRAPHY_MATS - 1);
    float t;
    
    BezierPoints points1, points2;
    if (wiggle) {
        points1 = {{{0,0.36},{0.3,2.235},{0.23,-0.28},{1,0.245}}};
        points2 = {{{0,0},{0.78,-0.316},{0.86,1.93},{1,1}}};
    } else {
        points1 = points2 = {{{0,0},{0.333,0.333},{0.666,0.666},{1,1}}};
    }
    
    for (int i = 0; i != MAX_HOMOGRAPHY_MATS; i++) {
        t = i * inc;
        make_homography(
                interpolate(translate_x[0], translate_x[1], bezier(points1, t)),
                interpolate(translate_y[0], translate_y[1], bezier(points2, t)),
                interpolate(scale_x[0], scale_x[1], t),
                interpolate(scale_y[0], scale_y[1], t),
                interpolate(rotate[0], rotate[1], t),
                interpolate(skew[0], skew[1], t),
                interpolate(project_x[0], project_x[1], t),
                interpolate(project_y[0], project_y[1], t),
                ref_point[0], ref_point[1],
                trajectory[i]);
    }

    auto elapsed = motionBlur(trajectory, MAX_HOMOGRAPHY_MATS, h_image, w, h, pixel_buf_bytes,
            steps_per_occu_block, h_blurred, viewport[0], viewport[1], exposure_mul);

    std::cout << output_path.str() << "\t" << elapsed << "\t" << out_opts.str() << std::endl; 
    int channels = 3;
    unsigned stride_in_bytes = viewport[0] * 3;
    stbi_write_png(output_path.str().c_str(), viewport[0], viewport[1], channels,
            h_blurred, stride_in_bytes);

    free(h_blurred);
    free(h_image);

    return 0;
}

