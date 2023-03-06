from mossetracker import MosseTracker
import cv2
#COLOR_BGR2GRAY
tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=True, hog=False, color=False)

tracker.initialize("./video_sequences/surfer_otb.mp4", useDetection=False)
print("hello")
coordinates= tracker.track(chooseNew=False)
print("hello")


#Hog works best for cards
#resnet is useless on card 