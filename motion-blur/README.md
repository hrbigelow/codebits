# A custom CUDA kernel for Motion Blur

This is a custom CUDA kernel for generating motion blur as defined by a list of
homography matrices.  A homography matrix defines a deformation of an input image
which maps straight lines to straight lines, but doesn't necessarily preserve angles,
distances or position.  Some examples of homographic transformations include:
displacement, rotation about any point, skew, scale (along any direction).  Another
way you can define a homography transformation is, choose any four points in the
image, and designate some destination position for that point.

To define motion blur, imagine mapping a sequence of slightly different homography
transformations, each to the same image, and then overlaying those images (averaging
the resulting color at each pixel).  A simple example is displacement in some
direction.  However, the displacement need not form a straight line path.  Or, if the
camera is moving forward, the transformations could be scaling in both x and y
directions holding some vanishing point stationary.  If the camera were rotated
within its focal plane (rare) then the image would appear spinning slightly, so
pixels further from the center would have more relative motion.

This kernel is able to simulate all of these situations by defining the homography
trajectory as a series of matrices.  Once defined, these matrices define a piecewise
linear motion path for each source pixel in the image.  The kernel samples locations
from this motion path at regular intervals.

## Building

```bash
git clone https://github.com/hrbigelow/codebits.git
cd codebits/motion-blur
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

./motion_blur
Usage: ./motion_blur <input_file> <trajectory_file> <viewport_width> <viewport_height> <steps_per_occu_block> <ref_point_x> <ref_point_y> <output_file>
./motion_blur input.png ../test/u-shaped.txt 3140 2060 40.0 1400 800 ~/blurred-input.png
```

See several example trajectories in `test` directory.  The format for these is:

    <num_matrices>
    mat1-row1
    mat1-row2
    mat1-row3
    mat2-row1
    ...
    ...

The remaining lines are rows of each matrix, each row containing three
space-separated numbers.

The trajectory file defines a piecewise-linear trajectory of `N-1` pieces for `N`
matrices.  The system can handle up to 64 matrices for a smoother trajectory.
Viewing this piecewise-linear trajectory as a single path, the system samples
`steps_per_occu_block * occu_blocks` nearly evenly-spaced points along this path.
This quantity is a cheap estimate of a given target location's total path length.
Note that certain trajectories like rotation result in different locations in the
image having different length trajectories.  The program thus samples longer
trajectories with more points so as to maintain fidelity.

The input image may be of any dimensions, and it will be treated as an infinite
canvas that wraps in both x and y directions.  This will of course cause unnatural
sampling artifacts where the blur crosses the image boundary.

All trajectories are defined relative to given reference point `ref_point_x,
ref_point_y`.  This doesn't affect translation trajectories, but it does affect
rotation and zoom, for example.  Mathematically, this is achieved by replacing every
given matrix `M` with:

$$
\begin{bmatrix}
1 & 0 & -x \\
0 & 1 & -y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
m_{00} & m_{01} & m_{02} \\
m_{10} & m_{11} & m_{12} \\
m_{20} & m_{21} & m_{22} \\
\end{bmatrix}
\begin{bmatrix}
1 & 0 & x \\
0 & 1 & y \\
0 & 0 & 1
\end{bmatrix}
$$



