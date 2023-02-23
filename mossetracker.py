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

    def __init__(self,gray_scale=False):
        self.is_gray_scale = gray_scale
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
        
        augmented_images = self.augmented_images(
            25, self.first_frame, (x, y, w, h))
        print(len(augmented_images))
        self.filter = self.create_filter(augmented_images)

    def read_first_frame(self):
        cap = cv2.VideoCapture(self.video_url)
        ret, frame = cap.read()
        self.first_frame = frame
        return cap

    def get_selected_region(self, frame):
        return get_selected_region_from_frame(frame)

    def track(self):
        n_times_occluded = [0]
        cap = cv2.VideoCapture(self.video_url)
        success, next_frame = cap.read()
        success = True
        peak = []
        ox, oy, ow, oh = self.selected_region
        frames = []
        fig, ax = plt.subplots()
        count = 1
        while success:
            
            print(self.selected_region)

            if not success:
                break
            # grey_im = Image.fromarray(next_frame).convert('L')
            # next_frame= np.array(grey_im)

            x, y, w, h = self.selected_region

            if self.is_gray_scale:
                log_img = np.log(cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY).astype(np.float64)+1)
                mean, std = np.mean(log_img), np.std(log_img)
                img_norm = (log_img - mean) / std
                F = np.fft.fft2(img_norm)
                output = self.apply_filter(F)
                # img = (cv2.cvtColor(crop_image(
                #     next_frame, x, y, w, h), cv2.COLOR_BGR2GRAY))
                # F = np.fft.fft2(img)
            else:
                log_B = np.log(cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h),cv2.COLOR_BGR2HSV)[:,:,0].astype(np.float64)+1)
                log_G = np.log(cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h),cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float64)+1)
                log_R = np.log(cv2.cvtColor(crop_image(
                    next_frame, x, y, w, h),cv2.COLOR_BGR2HSV)[:,:,2].astype(np.float64)+1)

        
                meanB, stdB = np.mean(log_B), np.std(log_B)
                meanG, stdG = np.mean(log_G), np.std(log_G)
                meanR, stdR = np.mean(log_R), np.std(log_R)
                
                norm_B = (log_B - meanB) / stdB
                norm_G = (log_G - meanG) / stdG
                norm_R = (log_R - meanR) / stdR

                # Cosine window 
                # window_col = np.hanning(w)
                # window_row = np.hanning(h)
                # col_mask, row_mask = np.meshgrid(window_col, window_row)
                # window = col_mask * row_mask
                # norm_B = norm_B * window
                # norm_G = norm_G * window
                # norm_R = norm_R * window
                

                F_Bi = np.fft.fft2(norm_B)
                F_Gi = np.fft.fft2(norm_G)
                F_Ri = np.fft.fft2(norm_R)
    
                output_B,output_G,output_R = self.apply_filter([F_Bi,F_Gi,F_Ri])

                output = output_B + output_G + output_R
                # plt.imshow(np.fft.ifft2(output).real)
                # plt.show()
                # break

            ux, uy = updateWindow(x, y, w, h, output, n_times_occluded)
            result_img_org = np.fft.ifft2(output).real
            # plt.imshow(result_img_org)
            # plt.show()

        

            self.selected_region = (ux, uy, w, h)
            

            # Display the image
            im = ax.imshow(next_frame,cmap="brg", animated=True)

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (ux, uy), w, h, linewidth=2, edgecolor='r', facecolor='none')
    

            # Add the patch to the Axes
            patch = ax.add_artist(rect)
            # ax.add_patch(rectOrg)
            frames.append([im, patch])
            # plt.show()
            if self.is_gray_scale:
                self.update_filter(F, output)
            else:
                self.update_filter_multi([F_Bi,F_Gi,F_Ri],[output_B,output_G,output_R])

            #print("Frame " + str(count) + " done")
            count +=1
            success, next_frame = cap.read()
        print("TIMES OCCLUDED",n_times_occluded, "/",count-1)
        ani = animation.ArtistAnimation(
            fig, frames, interval=30, blit=True, repeat_delay=0)
        plt.show()


    def apply_filter(self, frame):
        
        if self.is_gray_scale:
            return self.filter[0] * frame
        else:
            return [self.filter[0][0]*frame[0],self.filter[1][0]*frame[1],self.filter[2][0]*frame[2]]

    def augmented_images(sel, n_images, img, region):
        return get_augmented_images_cropped(n_images, img, region)

    def create_filter(self, array_of_images):
        return filterInit(array_of_images,gray_scale=self.is_gray_scale)

    def update_filter(self, F, G):
        self.filter = updateFilter(self.filter[1], self.filter[2], F, G)

    def update_filter_multi(self,F,G):
        F_B,F_G,F_R = F
        G_B,G_G,G_R = G

        filter_B = updateFilter(self.filter[0][1],self.filter[0][2],F_B,G_B)
        filter_G = updateFilter(self.filter[1][1],self.filter[1][2],F_G,G_G)
        filter_R = updateFilter(self.filter[2][1],self.filter[2][2],F_R,G_R)

        self.filter = [filter_B,filter_G,filter_R]

