from cv2 import imshow
import numpy as np
import cv2
import random
from scipy import misc
from matplotlib import pyplot as plt
from PIL import Image
import math


def crop_image(img,x,y,width,height):
    return img[y:y+height, x:x+width]

def rotate_image(image, angle, origin):
 
  rot_mat = cv2.getRotationMatrix2D(origin, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


"""
Should:
-Roate images
-Warp images
-Skew images

"""

def rnd(low,high):
    return random.uniform(low, high)



def get_augmented_images_cropped(number_of_images, img,crop_data):
    
    x,y,w,h = crop_data 
    origin = (x+ w//2,y+h//2)
    rows, cols = img.shape[:2]
    augmented_images_cropped = []
    

    img_cropped = crop_image(img,x,y,w,h)

    # Flip images veritically and horizontally
    flipped_image_horizontal = cv2.flip(img_cropped,1)
    flipped_image_vertical = cv2.flip(img_cropped,0)
    augmented_images_cropped.append(flipped_image_horizontal)
    augmented_images_cropped.append(flipped_image_vertical)


    #Blur
    ksize = (15, 15)
    blurred_image = cv2.blur(img_cropped,ksize)
    augmented_images_cropped.append(blurred_image)


    #Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(img_cropped, -1, kernel)
    augmented_images_cropped.append(sharpened_image)
    


    grey_im=Image.fromarray(img).convert('L')
    grey_im= np.array(grey_im)
    for i in range(number_of_images-4):
        # Randomly select a degree of rotation
        angles = [rnd(-45,-20),rnd(20,45)]
        
        idx = int(round(rnd(0,1)))
        angle = angles[idx]

        # Rotate
        rot_image = crop_image(rotate_image(img,angle,origin),x,y,w,h)
        augmented_images_cropped.append(rot_image)

        
        
        
        
    return augmented_images_cropped

