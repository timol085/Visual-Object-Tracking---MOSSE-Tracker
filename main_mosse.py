from mossetracker import MosseTracker

tracker = MosseTracker()

#redo after not found / out of bounds

tracker.initialize("rihanna.mp4", useDetection=True)

tracker.track()
