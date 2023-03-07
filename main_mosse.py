from matplotlib import pyplot as plt
import numpy as np
from mossetracker import MosseTracker
import cv2
import time

def rect_union(x1, y1, w1, h1, x2, y2, w2, h2):
    # Calculate the boundaries of each rectangle
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Calculate the boundaries of the union rectangle
    left = min(left1, left2)
    right = max(right1, right2)
    top = min(top1, top2)
    bottom = max(bottom1, bottom2)
    
    # Calculate the width and height of the union rectangle
    width = right - left
    height = bottom - top
    
    # Return the union rectangle's boundaries and dimensions
    return  width*height

def rect_intersection(x1, y1, w1, h1, x2, y2, w2, h2):
    # Calculate the boundaries of each rectangle
    left1, right1 = x1, x1 + w1
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Calculate the boundaries of the intersection rectangle
    left = max(left1, left2)
    right = min(right1, right2)
    top = max(top1, top2)
    bottom = min(bottom1, bottom2)
    
    # Check if the rectangles intersect, and return the intersection rectangle's boundaries and dimensions
    if left < right and top < bottom:
        width = right - left
        height = bottom - top
        return width*height
    else:
        return 0

def calculateiou(ourtracked, realtracked):
    iou = []
    for frame_idx, frame_data in enumerate(ourtracked):
        union = rect_union(ourtracked[frame_idx][0], ourtracked[frame_idx][1],ourtracked[frame_idx][2],ourtracked[frame_idx][3], 
                           realtracked[frame_idx][0],realtracked[frame_idx][1],realtracked[frame_idx][2],realtracked[frame_idx][3])
        intersection = rect_intersection(ourtracked[frame_idx][0], ourtracked[frame_idx][1],ourtracked[frame_idx][2],ourtracked[frame_idx][3], 
                           realtracked[frame_idx][0],realtracked[frame_idx][1],realtracked[frame_idx][2],realtracked[frame_idx][3])
        
        iou.append(intersection/union)
    return iou


#COLOR_BGR2GRAY
tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=True, hog=False, color=False)
images=[]
count=0

for i in range(1,10):
    images.append(cv2.imread(f"./video_sequences/imgf/000{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(10,100):
    images.append(cv2.imread(f"./video_sequences/imgf/00{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(100,893):
    images.append(cv2.imread(f"./video_sequences/imgf/0{i}.jpg", cv2.COLOR_BGR2RGB))
    

tracker.initialize(images, useDetection=False)
start_time = time.time()
coordinates= np.array(tracker.track(images,chooseNew=False))
print("--- %s resNet50 ---" % (time.time() - start_time))


File_data = np.loadtxt("groundtruth_rectf.txt", dtype=int)
iou= calculateiou(coordinates, File_data)
plt.plot(iou, label="ResNet50")

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=False, hog=False, color=False)
images=[]
count=0

for i in range(1,10):
    images.append(cv2.imread(f"./video_sequences/imgf/000{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(10,100):
    images.append(cv2.imread(f"./video_sequences/imgf/00{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(100,893):
    images.append(cv2.imread(f"./video_sequences/imgf/0{i}.jpg", cv2.COLOR_BGR2RGB))
    
    

tracker.initialize(images, useDetection=False)
start_time = time.time()
coordinates= np.array(tracker.track(images,chooseNew=False))
print("--- %s RGB ---" % (time.time() - start_time))


File_data = np.loadtxt("groundtruth_rectf.txt", dtype=int)
iou= calculateiou(coordinates, File_data)
plt.plot(iou, label="Only RGB")

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=False, hog=True, color=False)
images=[]
count=0

for i in range(1,10):
    images.append(cv2.imread(f"./video_sequences/imgf/000{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(10,100):
    images.append(cv2.imread(f"./video_sequences/imgf/00{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(100,893):
    images.append(cv2.imread(f"./video_sequences/imgf/0{i}.jpg", cv2.COLOR_BGR2RGB))
    
tracker.initialize(images, useDetection=False)

start_time = time.time()
coordinates= np.array(tracker.track(images,chooseNew=False))
print("--- %s HOG ---" % (time.time() - start_time))

File_data = np.loadtxt("groundtruth_rectf.txt", dtype=int)
iou= calculateiou(coordinates, File_data)
plt.plot(iou, label="HOG")

tracker = MosseTracker(cv2_color=cv2.COLOR_BGR2RGB,
                       resnet=False, hog=False, color=True)
images=[]
count=0

for i in range(1,10):
    images.append(cv2.imread(f"./video_sequences/imgf/000{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(10,100):
    images.append(cv2.imread(f"./video_sequences/imgf/00{i}.jpg", cv2.COLOR_BGR2RGB))
    
for i in range(100,893):
    images.append(cv2.imread(f"./video_sequences/imgf/0{i}.jpg", cv2.COLOR_BGR2RGB))
    
tracker.initialize(images, useDetection=False)
print("hello")

start_time = time.time()
coordinates= np.array(tracker.track(images,chooseNew=False))
print("--- %s HOG ---" % (time.time() - start_time))

File_data = np.loadtxt("groundtruth_rectf.txt", dtype=int)
iou= calculateiou(coordinates, File_data)
plt.plot(iou, label="Color names")


plt.legend()
plt.show()

