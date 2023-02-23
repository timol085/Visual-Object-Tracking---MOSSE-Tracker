from mossetracker import MosseTracker

tracker = MosseTracker(gray_scale=False)

#redo after not found / out of bounds

tracker.initialize("rihanna.mp4", useDetection=True)

tracker.track()
