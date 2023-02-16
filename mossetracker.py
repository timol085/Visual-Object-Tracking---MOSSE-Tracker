import cv2
from funcitons import get_selected_region_from_frame, get_augmented_images_cropped
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

    def initialize(self, video_url):
        self.video_url = video_url
        self.read_first_frame()
        x, y, w, h = self.get_selected_region(self.first_frame)
        self.selected_region = (x, y, w, h)
        print("InITAL", x, y, w, h)
        augmented_images = self.augmented_images(
            12, self.first_frame, (x, y, w, h))
        self.filter = self.create_filter(augmented_images)

    def read_first_frame(self):
        cap = cv2.VideoCapture(self.video_url)
        ret, frame = cap.read()
        self.first_frame = frame
        return cap

    def get_selected_region(self, frame):
        return get_selected_region_from_frame(frame)

    def track(self):
        cap = self.read_first_frame()
        success = True
        peak = []
        ox, oy, ow, oh = self.selected_region
        frames = []
        fig, ax = plt.subplots()

        while success:
            success, next_frame = cap.read()

            if not success:
                break
            # grey_im = Image.fromarray(next_frame).convert('L')

            # next_frame= np.array(grey_im)

            x, y, w, h = self.selected_region
            
            # np.fft.fft2(cv2.cvtColor(crop_image(next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY))
            F = np.fft.fft2(cv2.cvtColor(
                crop_image(next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY))
            output = self.apply_filter(F)
            ux, uy = updateWindow(x, y, w, h, output)
            result_img_org = np.fft.ifft2(output).real
            # plt.imshow(result_img_org)
            # plt.show()

            print("NEXT FRAME", (ux, uy, w, h))

            self.selected_region = (ux, uy, w, h)
            

            # Display the image
            im = ax.imshow(next_frame,animated=True)

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (ux, uy), w, h, linewidth=1, edgecolor='r', facecolor='none')
            print(len(next_frame))
            rectOrg = patches.Rectangle(
                (ox, oy), ow, oh, linewidth=1, edgecolor='g', facecolor='none')

            # Add the patch to the Axes
            patch = ax.add_artist(rect)
            #ax.add_patch(rectOrg)
            frames.append([im,patch])
            #plt.show()
            self.update_filter(F, output)
        ani = animation.ArtistAnimation(fig,frames,interval=30,blit=True,repeat_delay=0)
        plt.show()

    def apply_filter(self, frame):
        return self.filter[0] * frame

    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images)

    def update_filter(self, F, G):
        self.filter = updateFilter(self.filter[1], self.filter[2], F, G)

    def apply_filter_to_frame(self):
        pass
