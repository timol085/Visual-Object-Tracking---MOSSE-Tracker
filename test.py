import cv2
from matplotlib import pyplot as plt
from funcitons import get_augmented_images_cropped, crop_image

# Load the video

cap = cv2.VideoCapture("video.mp4")


# Get the first frame
ret, frame = cap.read()

# Display the frame and wait for user to select a region
crop_data = cv2.selectROI(frame)

x, y, w, h = crop_data


# Crop the selected region and save it as an image
crop = crop_image(frame, x, y, w, h)
cv2.imwrite("selected_region.jpg", crop)

augmented_images_cropped = get_augmented_images_cropped(12, frame, crop_data)

for cropped_img in augmented_images_cropped:
    plt.imshow(cropped_img, cmap=plt.get_cmap('gray'))
    plt.show()

# Clean up
cap.release()
cv2.destroyAllWindows()
