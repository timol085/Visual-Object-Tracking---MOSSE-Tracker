from mossetracker import MosseTracker
import cv2

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2HSV,resnet=True)

tracker.initialize("./video_sequences/test.mp4", useDetection=False)
print("hello")
tracker.track()
print("hello")