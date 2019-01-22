import cv2
import numpy as np
import matplotlib.pyplot as plt


def point_detection(img, kernel):
    image_1 = np.zeros(img.shape)
    image_2 = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            image_1[i][j] = abs(kernel[0][0]*img[i-1][j-1] + kernel[0][1]*img[i-1][j] + kernel[0][2]*img[i-1][j+1] \
                                + kernel[1][0] * img[i][j - 1] + kernel[1][1]*img[i][j] + \
                                kernel[1][2]*img[i][j+1] + kernel[2][0] * img[i+1][j - 1] + \
                                kernel[2][1]*img[i+1][j] + kernel[2][2]*img[i+1][j+1])

    cv2.imwrite("mask_output.jpg", image_1)
    max_val = np.amax(image_1)
    for i in range(0, image_1.shape[0]):
        for j in range(0, image_1.shape[1]):
            image_1[i][j] = (image_1[i][j]/max_val) * 255
            if image_1[i][j] >= 110:
                image_2[i][j] = 255
            else:
                image_2[i][j] = 0

    return image_2


def segment_detection(img):
    m1 = m2 = g1 = g2 = c1 = c2 = 0
    min_x = min_y = max_x = max_y = 0
    image_1 = np.zeros([img.shape[0], img.shape[1]])
    init_threshold = 203
    temp = 0
    x = []
    y = []
    while abs(temp - init_threshold) > 203:
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i][j] > init_threshold:
                    g1 = g1 + img[i][j]
                    c1 = c1 + 1
                else:
                    g2 = g2 + img[i][j]
                    c2 = c2 + 1
        m1 = g1 / c1
        m2 = g2 / c2
        temp = init_threshold
        init_threshold = (m1 + m2)/2

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > init_threshold:
                image_1[i][j] = img[i][j]

    img_result = image_1.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > init_threshold:
                x.append(j)
                y.append(i)

    x = np.array(x)
    y = np.array(y)
    min_x = np.amin(x)
    max_x = np.amax(x)
    min_y = np.amin(y)
    max_y = np.amax(y)
    pts = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    print("Coordinates of the boundary: ", pts)
    img_result[min_y:max_y, min_x] = 255
    img_result[min_y:max_y, max_x] = 255
    img_result[min_y, min_x:max_x] = 255
    img_result[max_y, min_x:max_x] = 255

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    result[min_y:max_y, min_x, 0] = 0
    result[min_y:max_y, min_x, 1] = 0
    result[min_y:max_y, min_x, 2] = 255
    result[min_y:max_y, max_x, 0] = 0
    result[min_y:max_y, max_x, 1] = 0
    result[min_y:max_y, max_x, 2] = 255
    result[min_y, min_y:max_x, 0] = 0
    result[min_y, min_x:max_x, 1] = 0
    result[min_y, min_x:max_x, 2] = 255
    result[max_y, min_x:max_x, 0] = 0
    result[max_y, min_x:max_x, 1] = 0
    result[max_y, min_x:max_x, 2] = 255
    return image_1, img_result, result


def histogram(image_1):
    values = [0] * 256
    for i in range(image_1.shape[0]):
        for j in range(image_1.shape[1]):
            values[image_1[i, j]] += 1
    points = []
    for i in range(0, 256):
        points.append(i)

    plt.bar(points, values)
    plt.ylabel('Intensity')
    plt.title("Histogram")
    plt.show()


def plot_point(img):
    image_1 = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if (img[j][i] != 0):
                print("Coordinates of the point: ", i, j)
                text = str(i) + " , " + str(j)
                cv2.putText(image_1, text, (i - 40, j + 40), font, fontScale, fontColor, lineType)
    return image_1


image_point = cv2.imread("input_images/turbine-blade.jpg", 0)
image_segment = cv2.imread("input_images/segment.jpg", 0)

print("Shape of point image: ", image_point.shape)

k = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

pd_image = point_detection(image_point, k)
plotted_point = plot_point(pd_image)
histogram(image_segment)
sg_image, sg_box_image, box_image = segment_detection(image_segment)

cv2.imwrite("Point.jpg", pd_image)
cv2.imwrite("Plotted_Point.jpg", plotted_point)
cv2.imwrite("Segment.jpg", sg_image)
cv2.imwrite("Segmented_Box.jpg", sg_box_image)
cv2.imwrite("Original_Box.jpg", box_image)
