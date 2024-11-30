import cv2
import numpy

# =====================================================
# bfs for filling labeling matrix

def is_valid(n, m, i, j):
    return n > i >= 0 and m > j >= 0

def bfs(matrix, label_matrix, start_i, start_j, label):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    queue = [(start_i, start_j)]

    while queue:
        front = queue.pop(0)
        cur_i, cur_j = front[0], front[1]

        for direction in directions:
            neighbour_i = cur_i + direction[0]
            neighbour_j = cur_j + direction[1]

            if not is_valid(len(matrix), len(matrix[0]), neighbour_i, neighbour_j):
                continue

            neighbour = int(matrix[neighbour_i][neighbour_j])

            if neighbour > 125 and label_matrix[neighbour_i][neighbour_j] == 0:
                label_matrix[neighbour_i][neighbour_j] = label
                queue.append((neighbour_i, neighbour_j))


# =====================================================

IMAGE_NUMBER = 1

while (photo := cv2.imread(f"multiple_objects_{IMAGE_NUMBER}.png")) is not None:  # Reading input images

    print(f"Processing multiple_objects_{IMAGE_NUMBER}.png")

    image_grayed = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    h, w = image_grayed.shape


    # filling the labeling matrix

    current_label = 1
    label_matrix = numpy.zeros((h, w))

    for i in range(h - 1):
        for j in range(w - 1):
            pixel = int(image_grayed[i][j])

            # if there is untouched pixel, start bfs

            if pixel > 125 and label_matrix[i][j] == 0:
                label_matrix[i][j] = current_label
                bfs(image_grayed, label_matrix, i, j, current_label)
                current_label += 1


    # calculating moments, center and box for every object (based on label)
    # analogically to the case with single object
    
    image = cv2.cvtColor(image_grayed, cv2.COLOR_GRAY2BGR)

    for INPUT_LABEL in range(1, current_label):

        image_temp = image_grayed.copy()

        for i in range(h - 1):
            for j in range(w - 1):
                label = label_matrix[i][j]
                if label != INPUT_LABEL:
                    image_temp[i][j] = 0
        
        moment0 = 0
        moment10 = 0
        moment01 = 0

        white_x = []
        white_y = []

        for i in range(h - 1):
            for j in range(w - 1):
                pixel = int(image_temp[i][j])
                moment0 += pixel
                moment10 += pixel * i
                moment01 += pixel * j
                if pixel > 125:
                    white_x.append(i)
                    white_y.append(j)
         
        area = moment0 / 255
        if area < 20:
            continue

        center_x = int(moment10 / moment0)
        center_y = int(moment01 / moment0)

        min_x, max_x = min(white_x), max(white_x)
        min_y, max_y = min(white_y), max(white_y)

        image = cv2.circle(img=image,
                        center=(center_y, center_x),
                        radius=5,
                        color=(0, 0, 255),
                        thickness=2)
        image = cv2.rectangle(img=image,
                            pt1=(min_y, min_x),
                            pt2=(max_y, max_x),
                            color=(0, 255, 0),
                            thickness=1)
        
    cv2.imwrite(f"multiple_regions_{IMAGE_NUMBER}.png", image)
    IMAGE_NUMBER += 1