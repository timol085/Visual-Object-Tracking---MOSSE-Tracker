import numpy as np
import cv2
from skimage.feature import hog

# Feature extraction for color and edge features 
def hog_extraction(normB, normG, normR):
    # HOG feature extraction
    fd_B, hog_image_B = hog(normB, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=True, channel_axis=-1, multichannel=False, feature_vector=True)
    fd_G, hog_image_G = hog(normG, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=True, channel_axis=-1, multichannel=False, feature_vector=True)
    fd_R, hog_image_R = hog(normR, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=True, channel_axis=-1, multichannel=False, feature_vector=True)
            
    # Combine feature descriptors for each channel
    fd = np.concatenate((fd_B, fd_G, fd_R))
    
    return fd
            
   # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
   # # ax1.axis('off')
   # # ax1.imshow(normR, cmap=plt.cm.gray)
   # ax1.set_title('Input image')
            
   # hog_image_rescaled = exposure.rescale_intensity(hog_image_R, in_range=(0, 10))
   # ax2.axis('off')
   # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
   # # ax2.set_title('Histogram of Oriented Gradients')
   # # plt.show()

