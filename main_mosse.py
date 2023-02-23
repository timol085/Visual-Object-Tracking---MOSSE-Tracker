from mossetracker import MosseTracker

tracker = MosseTracker(gray_scale=False)

tracker.initialize("Bolt.mp4")

tracker.track()
