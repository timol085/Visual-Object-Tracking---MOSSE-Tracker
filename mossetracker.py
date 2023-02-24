import cv2
from funcitons import get_selected_region_from_frame, get_augmented_images_cropped, get_detected_region_from_frame
from feature_extraction import hog_extraction
from filterInit import filterInit
from matplotlib import pyplot as plt
from funcitons import crop_image, preprocessing
from updateFilter import updateWindow, updateFilter
import matplotlib.patches as patches
import numpy as np
from matplotlib import animation


class MosseTracker:
    def __init__(self, cv2_color=cv2.COLOR_GRAY2BGR):
        self.color_mode = cv2_color
        self.video_url = None
        self.first_frame = None
        self.filter = None
        self.video = None
        self.selected_region = None
        self.useDetection = None

    def initialize(self, video_url, useDetection=False):
        self.video_url = video_url
        self.read_first_frame()
        self.useDetection = useDetection

        # do eiter detection or let user select regions
        if self.useDetection == False:
            x, y, w, h = self.get_selected_region(self.first_frame, False)

            self.selected_region = (x, y, w, h)
            augmented_images = self.augmented_images(
                12, self.first_frame, (x, y, w, h))
            self.filter = self.create_filter(augmented_images)
        else:
            returnvalue = self.get_selected_region(self.first_frame, True)
            if returnvalue != 1:
                x, y, w, h = returnvalue
                self.selected_region = (x, y, w, h)
                augmented_images = self.augmented_images(
                    12, self.first_frame, (x, y, w, h))
                self.filter = self.create_filter(augmented_images)
            else:
                # if it cannot find anything to detect, it will ask the user
                self.initialize(video_url)

    def read_first_frame(self):
        cap = cv2.VideoCapture(self.video_url)
        ret, frame = cap.read()
        self.first_frame = frame
        return cap

    def get_selected_region(self, frame, useDetection=False):
        if useDetection == False:
            return get_selected_region_from_frame(frame)
        else:
            return get_detected_region_from_frame(frame)

    def track(self):
        n_times_occluded = [0]
        cap = self.read_first_frame()
        image_width = self.first_frame.shape[1]
        image_height = self.first_frame.shape[0]
        print(image_width)
        print(image_height)
        success = True
        peak = []
        ox, oy, ow, oh = self.selected_region
        frames = []
        fig, ax = plt.subplots()
        count = 1
        while success:
            success, next_frame = cap.read()
            print(self.selected_region)
            x, y, w, h = self.selected_region
            img_color_mode = cv2.cvtColor(crop_image(next_frame, x, y, w, h), self.color_mode).astype(np.float64)
            _, _, num_channels = img_color_mode.shape
            if not success:
                break
            output = None
            all_F = []
            all_G = []
            for i in range(num_channels):
                i_img_norm = preprocessing(img_color_mode[:, :, i], width=w, height=h)
                i_F = np.fft.fft2(i_img_norm)
                i_output = self.apply_filter(i_F, i)
                if output is None:
                    output = i_output
                else:
                    output += i_output
                    print(output.shape)

                all_F.append(i_F)
                all_G.append(i_output)
                
            ux, uy = updateWindow(x, y, w, h, output, n_times_occluded)
            self.selected_region = (ux, uy, w, h)
            # Display the image
            im = ax.imshow(next_frame, cmap="brg", animated=True)
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (ux, uy), w, h, linewidth=1, edgecolor='r', facecolor='none')
            rectOrg = patches.Rectangle(
                (ox, oy), ow, oh, linewidth=1, edgecolor='g', facecolor='none')
            # Add the patch to the Axes
            patch = ax.add_artist(rect)
            frames.append([im, patch])
            self.update_filter(all_F,all_G)
            count += 1
            success, next_frame = cap.read()
        print("TIMES OCCLUDED", n_times_occluded, "/", count-1)
        ani = animation.ArtistAnimation(
            fig, frames, interval=30, blit=True, repeat_delay=0)

        plt.show()

    def apply_filter(self, frame, channel_idx):
        return self.filter[channel_idx][0] * frame
        
    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images, self.color_mode)
    
    def update_filter(self,F,G):
        filter_temp = []
        for idx,(f,g) in enumerate(zip(F,G)):
            self.filter[idx] = updateFilter(self.filter[idx][1],self.filter[idx][2], f, g)