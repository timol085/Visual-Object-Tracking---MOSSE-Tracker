import math
import cv2
from funcitons import get_selected_region_from_frame, get_augmented_images_cropped, get_detected_region_from_frame
from feature_extraction import hog_extraction, color_extraction
from filterInit import filterInit
from matplotlib import pyplot as plt
from funcitons import crop_image, preprocessing
from updateFilter import updateWindow, updateFilter
import matplotlib.patches as patches
import numpy as np
from matplotlib import animation
from resNet import resNet, getModel


class MosseTracker:
    def __init__(self, cv2_color=cv2.COLOR_GRAY2BGR, hog=False, resnet=False, color=False):
        self.color_mode = cv2_color
        self.video_url = None
        self.first_frame = None
        self.filter = None
        self.video = None
        self.selected_region = None
        self.useDetection = None
        self.HOG = hog
        self.ResNet = resnet
        self.color = color
        self.model = getModel()

    def initialize(self, video_url, useDetection=False):
        self.video_url = video_url
        self.read_first_frame()
        self.useDetection = useDetection
        # do eiter detection or let user select regions
        if self.useDetection == False:
            self.initialize_with_region()
        else:
            self.initialize_with_detection()

    def initialize_with_region(self):
        x, y, w, h = self.get_selected_region(self.first_frame, False)
        self.selected_region = (x, y, w, h)
        augmented_images = self.augmented_images(
            12, self.first_frame, (x, y, w, h))
        self.filter = self.create_filter(augmented_images)

    def initialize_with_detection(self):
        returnvalue = self.get_selected_region(self.first_frame, True)
        if returnvalue != 1:
            x, y, w, h = returnvalue
            self.selected_region = (x, y, w, h)
            augmented_images = self.augmented_images(
                12, self.first_frame, (x, y, w, h))
            self.filter = self.create_filter(augmented_images)
        else:
            # if it cannot find anything to detect, it will take next frame and try again
            cap = self.read_first_frame()
            canRead, self.first_frame = cap.read()
            while canRead:
                ret = self.get_selected_region(self.first_frame, True)
                if ret != 1:
                    canRead = False
                    x, y, w, h = ret
                    self.selected_region = (x, y, w, h)
                    augmented_images = self.augmented_images(
                        12, self.first_frame, (x, y, w, h))
                    self.filter = self.create_filter(augmented_images)
                canRead, self.first_frame = cap.read()

            print("Cant find")

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

        success = True
        peak = []

        if self.selected_region != None:
            ox, oy, ow, oh = self.selected_region
        else:
            success = False
        frames = []
        fig, ax = plt.subplots()
        count = 1

        while success:
            success, next_frame = cap.read()
            if not success:
                break
            x, y, w, h = self.selected_region

            if self.ResNet == False:
                img_color_mode = cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h), self.color_mode).astype(np.float64)
                if self.color == True:
                    img_color_mode = color_extraction(
                        img_color_mode, mode="probability")
                # ------------- For HOG multichannel --------------------#
                if self.HOG == True:
                    _, _, num_channels = img_color_mode.shape
                    for i in range(num_channels):
                        img_color_mode[:, :, i] = preprocessing(
                            img_color_mode[:, :, i], w, h)
                    hog_image_pre = cv2.resize(
                        img_color_mode, (w - w % 8, h-h % 8))
                    img_color_mode, _ = hog_extraction(hog_image_pre)
                    img_color_mode = np.squeeze(img_color_mode)
                    # img_color_mode = cv2.resize(
                    #     img_color_mode, (w, h))
                    _, _, hog_channels = img_color_mode.shape

                # ---------
                if len(img_color_mode.shape) == 2:
                    num_channels = 1
                else:
                    _, _, num_channels = img_color_mode.shape

            else:
                img_color_mode = resNet(crop_image(
                    next_frame, x, y, w, h), self.model)
                _, _, _, num_channels = img_color_mode.shape

            all_F = []
            all_G = []
            output = self.preprocess_and_calculate_filters(
                hog_channels if self.HOG == True else num_channels, img_color_mode, w, h, all_F, all_G)
            # For hog
            height_hog, width_hog = all_G[0].shape

            ux, uy = updateWindow(
                x, y, w, h, output, n_times_occluded, self.ResNet, self.HOG, width_hog, height_hog)
            self.selected_region = (ux, uy, w, h)
            # Display the image
            im = ax.imshow(next_frame, cmap="brg", animated=True)
            self.draw_rectangle(ux, uy, w, h, ox, oy, ow, oh, ax, frames, im)
            self.update_filter(all_F, all_G)
            count += 1
            success, next_frame = cap.read()
        print("TIMES OCCLUDED", n_times_occluded, "/", count-1)
        ani = animation.ArtistAnimation(
            fig, frames, interval=30, blit=True, repeat_delay=0)

        plt.show()

    def apply_filter(self, frame, channel_idx):
        print("index", channel_idx)
        print("self.filter shape", len(self.filter))
        return self.filter[channel_idx][0] * frame

    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images, self.color_mode, self.ResNet, self.HOG, self.color, self.model)

    def update_filter(self, F, G):
        filter_temp = []
        for idx, (f, g) in enumerate(zip(F, G)):
            self.filter[idx] = updateFilter(
                self.filter[idx][1], self.filter[idx][2], f, g, self.ResNet)

    def preprocess_and_calculate_filters(self, num_channels, img_color_mode, w, h, all_F, all_G):
        output = None

        for i in range(num_channels):
            if self.ResNet == False and self.color == False:
                # ------------For HOG multichannel ----------------#
                if self.HOG == True:

                    i_img_norm = img_color_mode[:, :, i]
                # ----
                else:
                    if len(img_color_mode.shape) == 2:
                        i_img_norm = preprocessing(
                            img_color_mode, width=w, height=h)
                    else:
                        i_img_norm = preprocessing(
                            img_color_mode[:, :, i], width=w, height=h)
            elif self.ResNet == True:
                i_img_norm = img_color_mode[0, :, :, i]
            else:
                i_img_norm = img_color_mode[:, :, i]

            # ----------------For HOG in all seperate RGB channels------------------------#
            # if self.HOG == True:
            #     i_img_norm_hog, _ = hog_extraction(i_img_norm)
            #     i_img_norm_hog = np.squeeze(i_img_norm_hog)
            #     _, _, num_channels_hog = i_img_norm_hog.shape

            #     for j in range(num_channels_hog):
            #         i_F = np.fft.fft2(i_img_norm_hog[:,:,j])
            #         i_output = self.apply_filter(i_F, i*num_channels_hog+ j)
            #         if output is None:
            #             output = i_output
            #         else:
            #             output += i_output
            #             print(output.shape)

            #         all_F.append(i_F)
            #         all_G.append(i_output)

            # else:
            # ----

            i_F = np.fft.fft2(i_img_norm)
            i_output = self.apply_filter(i_F, i)
            if output is None:
                output = i_output
            else:
                output += i_output
                print(output.shape)

            all_F.append(i_F)
            all_G.append(i_output)
        return output

    def draw_rectangle(self, ux, uy, w, h, ox, oy, ow, oh, ax, frames, im):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (ux, uy), w, h, linewidth=1, edgecolor='r', facecolor='none')
        rectOrg = patches.Rectangle(
            (ox, oy), ow, oh, linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        patch = ax.add_artist(rect)
        frames.append([im, patch])
