from cv2 import imshow
import numpy as np
import cv2
import random
from scipy import misc
from matplotlib import pyplot as plt
from PIL import Image


def transform_images(number_of_images, img):
    rows, cols = img.shape[:2]
    transformed_images = []
    # this dont work, change it

    grey_im = Image.fromarray(img).convert('L')
    grey_im = np.array(grey_im)
    for i in range(number_of_images):
        # Randomly select a degree of rotation
        angle = random.uniform(-90, 90)
        rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # Randomly select a scaling factor
        scale = random.uniform(0.5, 1.5)
        sc_mat = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)

        # Randomly select a translation factor
        tx = random.uniform(-cols * 0.2, cols * 0.2)
        ty = random.uniform(-rows * 0.2, rows * 0.2)
        tr_mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

        # Combine the transformation matrices
        combined_mat = rot_mat  # rot_mat @ sc_mat.transpose() @ tr_mat

        # Perform the affine transformation
        transformed_image = cv2.warpAffine(grey_im, combined_mat, (cols, rows))

        transformed_image = Image.fromarray(transformed_image).convert('L')
        transformed_image = np.array(transformed_image)
        plt.imshow(transformed_image, cmap=plt.get_cmap('gray'))
        plt.show()

    return transformed_images
