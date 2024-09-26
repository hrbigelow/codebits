# A custom CUDA kernel for Motion Blur

This is a custom CUDA kernel for generating motion blur as defined by any combination
of translation, rotation, skew, projection and scaling.  Internally, all of these
aspects of an image deformation are expressed by a homography matrix, and a
*trajectory* consists of a sequence of such homography matrices.

In this demo, each aspect is provided as a *start* and *end* value with defaults set
to the non-effect value (zero for translation, rotation, skew, and projection, and
one for scale).

More flexible sets of motion can be produced for the CUDA kernel by manually
providing a homography matrix array, but that is not provided by this tool.  However,
you can combine any of these aspects.

In general, the computation takes longer when sampling density (see
`--steps-per-block`) is higher, or when the path a given pixel travels is longer.
For example, when applying rotation about some point, pixels far from that point will
travel farther during the trajectory and thus take longer to compute.

The kernel is quite performant however, computing many of these images in under 0.2
seconds (some very large, 6000 x 4000 pixel images) depending on the task.

## Building

```bash
git clone https://github.com/hrbigelow/codebits.git
cd codebits/motion-blur
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```


# Running 

```bash
$ ./motion_blur
input_image: 1 argument(s) expected. 0 provided.
Usage: motion_blur [--help] [--version] [--viewport VAR...] [--steps-per-block VAR] [--reference-point VAR...] [--translate-x VAR...] [--translate-y VAR...] [--scale-x VAR...] [--scale-y VAR...] [--rotate VAR...] [--skew VAR...] [--project-x VAR...] [--project-y VAR...] [--pixel-buf-bytes VAR] input_image output_image

Positional arguments:
  input_image               source PNG image file
  output_image              Output PNG filename

Optional arguments:
  -h, --help                shows help message and exits
  -v, --version             prints version information and exits
  -vp, --viewport           output viewport (width, height) (default (0, 0) will be set to input image size) [nargs=0..2] [default: {0 0}]
  -sb, --steps-per-block    Trajectory sampling density in steps per occupied (32 x 32) pixel block [nargs=0..1] [default: 10]
  -ref, --reference-point   Reference point (x, y) about which the homographies are taken [nargs=0..2] [default: {250 125}]
  -tx, --translate-x        start and end horizontal translation in pixels [nargs=0..2] [default: {0 0}]
  -ty, --translate-y        start and end vertical translation in pixels [nargs=0..2] [default: {0 0}]
  -sx, --scale-x            start and end horizontal scale factor [nargs=0..2] [default: {1 1}]
  -sy, --scale-y            start and end vertical scale factor [nargs=0..2] [default: {1 1}]
  -r, --rotate              start and end rotation angle in degrees [nargs=0..2] [default: {0 0}]
  -sk, --skew               start and end skew [nargs=0..2] [default: {0 0}]
  -px, --project-x          start and end horizontal projection [nargs=0..2] [default: {0 0}]
  -py, --project-y          start and end vertical projection [nargs=0..2] [default: {0 0}]
  -pbuf, --pixel-buf-bytes  size in bytes for shared memory pixel buffer [nargs=0..1] [default: 45056]
  -wig, -wiggle             If present, produce a wiggled path for translation
```

# Examples

![krisof](./img/krisof-small.jpg)

Original [photo](https://www.pexels.com/photo/scenic-view-of-night-sky-1252871/) by
Hristo Fidanov 

![krisof-rot.png](./img/krisof-rot.jpg)

Photo with rotation blur applied about the point (300, 350) through degrees -1.5 to
1.5.

    motion_blur krisof-small.jpg krisof-rot.png \
        --rotate -1.5 1.5 \
        --reference-point 300 350 \
        --steps-per-block 10

![krisof-wiggle.png](./img/krisof-wiggle.jpg)

Photo with translation displacements of (-5, -3) through (5, 3) following a wiggled path.

    motion_blur krisof-small.jpg krisof-wiggle.png \
        --translate-x -5 5 \
        --translate-y -3 3 \
        --steps-per-block 10 \
        --wiggle

![dreamsky](./img/dreamsky-small.jpg)

The original
[photo](https://www.pexels.com/photo/photography-of-fireworks-display-790916/) known
as Dream Sky.

![dreamsky-trans](./img/dreamsky-trans.jpg)

Dream Sky photo with translation blur (but no wiggle)

    motion_blur dreamsky-small.jpg dreamsky-trans.png \
        --translate-x -5 5 \
        --translate-y -3 3 \
        --steps-per-block 10

![mikebirdy](./img/mikebirdy-small.jpg)

The original MikeBirdy
[photo](https://www.pexels.com/photo/pathway-between-trees-towards-house-126271/)
provided by [pexels.com](pexels.com).

![mikebirdy](./img/mikebirdy-scale.jpg)

Scale blur - starting with the original (1, 1) scale and magnifying to (1.03, 1.03)
(in x and y directions) using point (450, 300) as the origin.

    motion_blur mikebirdy-small.jpg mikebirdy-scale.png \
        --reference-point 450 300 \
        --scale-x 1.0 1.03 \
        --scale-y 1.0 1.03 \
        --steps-per-block 10

