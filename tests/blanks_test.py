import cv2
import numpy as np

path = "test.png"

image = cv2.imread(path)
greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(greyscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

scale_percent = 500  # percent of original size
width = int(thresh.shape[1] * scale_percent / 100)
height = int(thresh.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

kernel = np.ones((10, 10), np.uint8)
dilated_image = cv2.dilate(thresh, kernel)

lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(dilated_image)[0]
print(len(list(lines)) / 2)
drawn_img = lsd.drawSegments(dilated_image, lines)

cv2.imshow("Image", drawn_img)
cv2.waitKey()
