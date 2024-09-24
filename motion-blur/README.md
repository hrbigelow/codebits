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

## Command-Line

```bash
$ motion_blur
Usage: motion_blur \
    <input_file> <trajectory_file> \
    <viewportWidth> <viewportHeight> \
    <steps_per_occu_block> <output_file>
```




The user inputs a PNG image (any dimensions are acceptable), a
sequence of N (<= 64) homography matrices, and a parameter `steps_per_occu_block`
which gives the number of steps 
