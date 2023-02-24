import cv2
from funcitons import get_selected_region_from_frame, get_augmented_images_cropped, get_detected_region_from_frame
from feature_extraction import hog_extraction
from filterInit import filterInit
from matplotlib import pyplot as plt
from funcitons import crop_image, preprocessing
from get_peak_and_psr import get_peak_and_psr
from updateFilter import updateWindow, updateFilter
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from matplotlib import cm
from matplotlib import animation


# Global variables
global GRAY_SCALE
global RGB
global HOG
global RESNET
GRAY_SCALE = 0
RGB = 1
HOG = 2
RESNET = 3


class MosseTracker:
    def __init__(self, chosen_channel=GRAY_SCALE):
        self.channel = chosen_channel
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

            if not success:
                break
            x, y, w, h = self.selected_region

            if self.channel == GRAY_SCALE:
                img_gray = cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY).astype(np.float64)

                img_norm = preprocessing(img_gray, width=w, height=h)
                F = np.fft.fft2(img_norm)
                output = self.apply_filter(F)
                # img = (cv2.cvtColor(crop_image(
                #     next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY))
                # F = np.fft.fft2(img)
            else:
                # preprocessing fixat
                img_B = cv2.cvtColor(crop_image(next_frame, x, y, w, h), cv2.COLOR_BGR2HSV)[
                    :, :, 0].astype(np.float64)
                img_G = cv2.cvtColor(crop_image(next_frame, x, y, w, h), cv2.COLOR_BGR2HSV)[
                    :, :, 1].astype(np.float64)
                img_R = cv2.cvtColor(crop_image(next_frame, x, y, w, h), cv2.COLOR_BGR2HSV)[
                    :, :, 2].astype(np.float64)

                norm_B = preprocessing(img_B, width=w, height=h)
                norm_G = preprocessing(img_G, width=w, height=h)
                norm_R = preprocessing(img_R, width=w, height=h)

                fd, fd_B, fd_G, fd_R, imgB, imgG, imgR = hog_extraction(
                    norm_B, norm_G, norm_R)

                F_Bi = np.fft.fft2(imgB)
                F_Gi = np.fft.fft2(imgG)
                F_Ri = np.fft.fft2(imgR)

                output_B, output_G, output_R = self.apply_filter(
                    [F_Bi, F_Gi, F_Ri])

                output = output_B + output_G + output_R
                # plt.imshow(np.fft.ifft2(output).real)
                # plt.show()
                # break

            ux, uy = updateWindow(x, y, w, h, output, n_times_occluded)
            result_img_org = np.fft.ifft2(output).real

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
            # ax.add_patch(rectOrg)
            frames.append([im, patch])
            # plt.show()
            if self.channel == GRAY_SCALE:
                self.update_filter(F, output)
            else:
                self.update_filter_multi([F_Bi, F_Gi, F_Ri], [
                                         output_B, output_G, output_R])

            # print("Frame " + str(count) + " done")
            count += 1
            success, next_frame = cap.read()
        print("TIMES OCCLUDED", n_times_occluded, "/", count-1)
        ani = animation.ArtistAnimation(
            fig, frames, interval=30, blit=True, repeat_delay=0)

        plt.show()

    def apply_filter(self, frame):

        if self.channel == GRAY_SCALE:
            return self.filter[0] * frame
        else:
            return [self.filter[0][0]*frame[0], self.filter[1][0]*frame[1], self.filter[2][0]*frame[2]]

    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images, channel=GRAY_SCALE)

    def update_filter(self, F, G):
        self.filter = updateFilter(self.filter[1], self.filter[2], F, G)

    def update_filter_multi(self, F, G):
        F_B, F_G, F_R = F
        G_B, G_G, G_R = G

        filter_B = updateFilter(self.filter[0][1], self.filter[0][2], F_B, G_B)
        filter_G = updateFilter(self.filter[1][1], self.filter[1][2], F_G, G_G)
        filter_R = updateFilter(self.filter[2][1], self.filter[2][2], F_R, G_R)

        self.filter = [filter_B, filter_G, filter_R]
