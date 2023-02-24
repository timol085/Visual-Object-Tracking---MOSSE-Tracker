from mossetracker import MosseTracker
import cv2

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2HSV,resnet=True)

tracker.initialize("./video_sequences/rihanna.mp4", useDetection=True)
print("hello")
tracker.track()
print("hello")