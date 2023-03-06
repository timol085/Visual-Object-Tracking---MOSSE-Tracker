from mossetracker import MosseTracker
import cv2

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=False, hog=False, color=True)

tracker.initialize("./video_sequences/surfer_otb.mp4", useDetection=False)
print("hello")
tracker.track()
print("hello")
