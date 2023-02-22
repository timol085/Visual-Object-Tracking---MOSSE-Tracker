from mossetracker import MosseTracker

tracker = MosseTracker(gray_scale=False)

tracker.initialize("surfer.mp4")

tracker.track()
