import cv2
from funcitons import get_selected_region_from_frame, get_augmented_images_cropped, get_detected_region_from_frame
from filterInit import filterInit
from matplotlib import pyplot as plt
from funcitons import crop_image
from get_peak_and_psr import get_peak_and_psr
from updateFilter import updateWindow, updateFilter
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from matplotlib import cm
from matplotlib import animation


class MosseTracker:

    def __init__(self):
        self.video_url = None
        self.first_frame = None
        self.filter = None
        self.video = None
        self.selected_region = None
        self.useDetection= None

    def initialize(self, video_url, useDetection=False):
        self.video_url = video_url
        self.read_first_frame()
        self.useDetection=useDetection
        
        #do eiter detection or let user select regions
        if self.useDetection==False:
            x, y, w, h = self.get_selected_region(self.first_frame, False)
                
            self.selected_region = (x, y, w, h)
            augmented_images = self.augmented_images(
                12, self.first_frame, (x, y, w, h))
            self.filter = self.create_filter(augmented_images)
        else:
            returnvalue=self.get_selected_region(self.first_frame, True)
            if returnvalue!=1:
                x, y, w, h = returnvalue      
                self.selected_region = (x, y, w, h)
                augmented_images = self.augmented_images(
                    12, self.first_frame, (x, y, w, h))
                self.filter = self.create_filter(augmented_images)
            else: 
                #if it cannot find anything to detect, it will ask the user
                self.initialize(video_url)
                

    def read_first_frame(self):
        cap = cv2.VideoCapture(self.video_url)
        ret, frame = cap.read()
        self.first_frame = frame
        return cap

    def get_selected_region(self, frame, useDetection=False):
        if useDetection==False:
            return get_selected_region_from_frame(frame)
        else:
            return get_detected_region_from_frame(frame)
        
    def track(self):
        cap = self.read_first_frame()
        image_width= self.first_frame.shape[1]
        image_height= self.first_frame.shape[0]
        print(image_width)
        print(image_height)
        success = True
        peak = []
        ox, oy, ow, oh = self.selected_region
        frames = []
        fig, ax = plt.subplots()

        while success:
            success, next_frame = cap.read()
            if not success:
                break
            x, y, w, h = self.selected_region

            log_img = np.log(cv2.cvtColor(crop_image(
                next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY).astype(np.float64)+1)
            mean, std = np.mean(log_img), np.std(log_img)
            img_norm = (log_img - mean) / std
            F = np.fft.fft2(img_norm)

            img = (cv2.cvtColor(crop_image(
                next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY))
            F = np.fft.fft2(img)

            output = self.apply_filter(F)
            ux, uy = updateWindow(x, y, w, h, output)
            result_img_org = np.fft.ifft2(output).real

            self.selected_region = (ux, uy, w, h)

            # Display the image
            im = ax.imshow(
                (cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)), animated=True)
            # Create a Rectangle patch

            rect = patches.Rectangle(
                (ux, uy), w, h, linewidth=1, edgecolor='r', facecolor='none')
            rectOrg = patches.Rectangle(
                (ox, oy), ow, oh, linewidth=1, edgecolor='g', facecolor='none')
            # Add the patch to the Axes
            patch = ax.add_artist(rect)
            # ax.add_patch(rectOrg)
            frames.append([im, patch])
            # plt.show()
            self.update_filter(F, output)
        ani = animation.ArtistAnimation(
            fig, frames, interval=30, blit=True, repeat_delay=0)

           
        plt.show()

    def apply_filter(self, frame):
        return self.filter[0] * frame

    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images)

    def update_filter(self, F, G):
        self.filter = updateFilter(self.filter[1], self.filter[2], F, G)
