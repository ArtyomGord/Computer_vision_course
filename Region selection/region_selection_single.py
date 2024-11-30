import cv2
import numpy

# reading image

photo = cv2.imread("single_object.png")
grayed = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

h, w = grayed.shape


moment0 = 0
moment10 = 0
moment01 = 0

white_x = []
white_y = []

# calculating moments

for i in range(h - 1):
    for j in range(w - 1):
      pixel = int(grayed[i][j])
      moment0 += pixel
      moment10 += pixel * i
      moment01 += pixel * j
      if pixel > 125:
        white_x.append(i)
        white_y.append(j)

# calculating center of an object

area = moment0 / 255
center_x = int(moment10 / moment0)
center_y = int(moment01 / moment0)

min_x, max_x = min(white_x), max(white_x)
min_y, max_y = min(white_y), max(white_y)

# drawing bounding box and circle at center

image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
image = cv2.circle(img=image,
                   center=(center_y, center_x),
                   radius=5,
                   color=(0, 0, 255),
                   thickness=2)
final = cv2.rectangle(img=image,
                      pt1=(min_y, min_x),
                      pt2=(max_y, max_x),
                      color=(0, 255, 0),
                      thickness=1)

cv2.imwrite("single_region.png", final)