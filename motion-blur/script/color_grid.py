from PIL import Image, ImageDraw
from random import randint
import random
import fire

# Function to generate a random color
def random_color(min_sum=200, max_sum=765):
    r = randint(max(min_sum - 510, 0), 255)
    g = randint(max(min_sum - 255 - r, 0), 255)
    b = randint(max(min_sum - r - g, 0), 255)
    if r + g + b < min_sum:
        print(f'Failed min_sum test: {r}, {g}, {b}')
    return r, g, b 

# Image size and grid configuration
def main(image_width, image_height, grid_size, num_dots, radius, background_color, out_file):
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    for i in range(num_dots):
        x = randint(0, image_width)
        y = randint(0, image_height)
        if i % 2 == 0:
            x -= (x % grid_size)
        else:
            y -= (y % grid_size)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=random_color())

    image.save(out_file)

if __name__ == '__main__':
    fire.Fire(main)

