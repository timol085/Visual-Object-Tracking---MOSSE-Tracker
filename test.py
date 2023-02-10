import cv2
from funcitons import transform_images

# Load the video
cap = cv2.VideoCapture("d.mp4")

# Get the first frame
ret, frame = cap.read()

# Display the frame and wait for user to select a region
r = cv2.selectROI(frame)

print(r[0])
print(r[1])
print(r[2])
print(r[3])
# Crop the selected region and save it as an image
crop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imwrite("selected_region.jpg", crop)
print(crop.shape)
transform_images(5, crop)

# Clean up
cap.release()
cv2.destroyAllWindows()
