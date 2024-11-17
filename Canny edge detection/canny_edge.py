import cv2
import numpy as np
import math

# hyperparameters for weak/strong edge detecting
low_thr = 5
upper_thr = 10


IMAGE_NUMBER = 1

while (image := cv2.imread(f"input_image_{IMAGE_NUMBER}.png")) is not None:  # Reading input images

    print(f"Processing input_image_{IMAGE_NUMBER}.png")

    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_padded = np.pad(image_grayscale, ((1, 1), (1, 1)), 'constant', constant_values=255)  # 1 pixel padding
    blurred = cv2.GaussianBlur(image_padded, (5, 5), 0)  # 5x5 kernel gaussian blur
    h, w = blurred.shape

    final = np.zeros((h, w), dtype=np.uint8)


    # calculating gradient angle and intensivity for each pixel 

    grad_angle = np.zeros((h, w))
    grad_intense = np.zeros((h, w))
    angles_list = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    for i in range(1, h-1):
        for j in range(1, w-1):
            grad_x = (int(blurred[i][j + 1]) -  int(blurred[i][j - 1])) / 2
            grad_y = (int(blurred[i + 1][j]) -  int(blurred[i - 1][j])) / 2

            grad_intense[i][j] = math.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_angle[i][j] = round(math.degrees(math.atan2(grad_y, grad_x)))


    # deciding the direction of the gradient based on angle

    for i in range(1, h-1):
        for j in range(1, w-1):

            pixel_angle = grad_angle[i][j]
            pixel_intense = grad_intense[i][j]
            angles_diff = [0 for _ in range(9)]

            for idx, angle in enumerate(angles_list):
                angles_diff[idx] = abs(pixel_angle - angle)

            angle = min(angles_diff)
            index =  angles_diff.index(angle)
            angle = angles_list[index]

            if angle == -180 or angle == 180 or angle == 0:
                pixel_1 = grad_intense[i][j - 1]
                pixel_2 = grad_intense[i][j + 1]
            elif angle == 45 or angle == -135:
                pixel_1 = grad_intense[i - 1][j - 1]
                pixel_2 = grad_intense[i + 1][j + 1]
            elif angle == 90 or angle == -90:
                pixel_1 = grad_intense[i - 1][j]
                pixel_2 = grad_intense[i + 1][j]
            elif angle == -45 or angle == 135:
                pixel_1 = grad_intense[i + 1][j - 1]
                pixel_2 = grad_intense[i - 1][j + 1]

            # checking if the pixel is considered as edge

            on_edge = pixel_intense > pixel_1 and pixel_intense > pixel_2

            if on_edge and pixel_intense > low_thr:
                if pixel_intense < upper_thr:
                    final[i][j] = 125  # weak edge
                else:
                    final[i][j] = 255  # strong edge
            else:
                final[i][j] = 0

    cv2.imwrite(f"edges_{IMAGE_NUMBER}.png", final)
    IMAGE_NUMBER += 1