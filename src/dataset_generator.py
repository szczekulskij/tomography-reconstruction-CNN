import numpy as np
import pandas as pd
from random import randint
import math
from PIL import Image, ImageDraw
from PIL import ImagePath 
from imutils import rotate
import matplotlib.pyplot as plt




def convert_to_binary_img(img):
    after_conversion = []
    for x in np.array(img):
        x_dim = []
        for y in x:
            y_dim = []
            if not np.all((y == 0)):
                x_dim.append(1)
            else : 
                x_dim.append(0)
        after_conversion.append(x_dim)
    return np.array(after_conversion)
        
def generate_polygon(
    side = 2, # nr of sides/2
    rotated = True,
    randomly_placed = True,
    noise = False,
    pct_size_range = [20,80], # What % size of the picture can the polygon take ?
    img_size = 64):   

    img = None
    while img is None:
        img, rotation_angle = generate_polygon_subroutine(side = side,
                                        rotated = rotated,
                                        randomly_placed = randomly_placed,
                                        noise = noise,
                                        pct_size_range = pct_size_range, 
                                        img_size = img_size)
    return img, rotation_angle




def generate_polygon_subroutine(
    side = 4, # nr of sides/2
    rotated = True,
    randomly_placed = True,
    noise = False,
    pct_size_range = [20,80], # What % size of the picture can the polygon take ?
    img_size = 128):        

    PADDING = 1
    side_size_range = [round(i/100 * img_size/2) for i in pct_size_range]
    assert len(side_size_range) == 2
    assert side in [4,5,6]
    side_size = randint(*side_size_range)


    coordinates = [
        (round((math.cos(th) + 1) * side_size),
        round((math.sin(th) + 1) * side_size))
        for th in [i * (2 * math.pi) / side for i in range(side)]
        ]  

    # Re-centre or randomly place the image
    if randomly_placed:
        x_offset = randint(PADDING, img_size - side_size * 2 - PADDING)
        y_offset = randint(PADDING, img_size - side_size * 2 - PADDING)
    else:
        x_offset = img_size/2 - side_size
        y_offset = img_size/2 - side_size
    coordinates_before = coordinates
    coordinates = [(x + x_offset, y + y_offset)for x,y in coordinates]




    image = ImagePath.Path(coordinates).getbbox()  
    img = Image.new("RGB", (img_size, img_size)) 
    img1 = ImageDraw.Draw(img)  
    img1.polygon(coordinates, fill = 200)

    if type(rotated) == int:
        rotation_angle = rotated
    else : 
        rotation_angle = randint(0,90) if rotated else 0
    img = img.rotate(rotation_angle)

    img = convert_to_binary_img(img)


    ### Double check that the image hasn't been cut!
    existing_area = img.sum()
    if side == 4:
        should_be_area = side_size*side_size
        # ratio is ~2
    elif side ==5 :
        should_be_area = 1/4 * math.sqrt(5 * (5 + 2 * math.sqrt(5))) * side_size * side_size
        # ratio is ~1.40
    elif side == 6:
        should_be_area = 3 * math.sqrt(3)/2 * side_size * side_size
        # ratio around 1

    ratio = existing_area/should_be_area

    if side == 4 and ratio < 1.98:
        return None, rotation_angle
    elif side == 5 and ratio <1.38:
        return None, rotation_angle
    elif side == 6 and ratio < 0.98:
        return None, rotation_angle
    else :
        return img, rotation_angle


def calculate_angles(side, rotation_angle, binary = True):
    if side == 4:
        # start_angle = 45
        angles = [45, 135]
    elif side == 5:
        # start_angle = 18
        angles = [18, 90, 162]
    elif side == 6:
        # start_angle = 0
        angles = [0, 60, 120, 180]

    post_rotation_angles = [round((i + rotation_angle) % 180) for i in angles]
    binary_angles = [0] * 180
    for angle_index in post_rotation_angles:
        binary_angles[angle_index] = 1

    if binary:
        return binary_angles
    else : 
        return post_rotation_angles
