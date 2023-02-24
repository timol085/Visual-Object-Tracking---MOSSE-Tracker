from mossetracker import MosseTracker
import cv2

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2HSV)

tracker.initialize("./video_sequences/surfer.mp4")

tracker.track()
