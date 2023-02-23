from mossetracker import MosseTracker

global GRAY_SCALE
global RGB
global HOG
global RESNET
GRAY_SCALE = 0
RGB = 1
HOG = 2
RESNET = 3

tracker = MosseTracker(RGB)

tracker.initialize("Bolt.mp4")

tracker.track()
