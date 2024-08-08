'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!

def merge_with_median(arr):
    if len(arr) == 0:
        return arr

    merged = []
    start = 0

    while start < len(arr):
        end = start
        while end + 1 < len(arr) and arr[end + 1] - arr[end] == 1:
            end += 1

        if start == end:
            merged.append(arr[start])
        else:
            merged.append(int(np.median(arr[start:end + 1])))
        start = end + 1

    return np.array(merged)


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    binarized_img = np.where(img > 115, 255, 0).astype(np.uint8)
    smoothed_img = cv2.GaussianBlur(binarized_img, (5, 5), 0)
    return smoothed_img


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_image = np.zeros_like(img)
    cv2.drawContours(contour_image, [largest_contour], -1, (255, 255, 255), 2)
    return contour_image

def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    # Get the column at position x
    column = contour_img[:, x]

    # Find the indices (y-values) where the column intersects the contours (contour pixels are black (0))
    intersections = np.where(column == 255)[0]

    intersections = merge_with_median(intersections)




    while len(intersections) > 6:
        if abs(intersections[0] - 0) < abs(intersections[-1] - contour_img.shape[0]):
            intersections = intersections[1:]
        else:
            intersections = intersections[:-1]

    return intersections
def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''

    # line y = (m) * x + c
    slope = (y2 - y1) / (x2 - x1)
    #find value of c
    c = y1 - slope * x1
    # find value of coordinate where contour is black

    for i in range(x2 + 1, img.shape[1]):
        y = int(slope * i + c)

        if img[y][i] == 255:
            break
    return (y, i)

def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''

    slope = (k3[0] - k1[0]) / (k3[1] - k1[1])
    c1 = k3[0] - slope * k3[1]

    perpendicular_slope = -1 / slope
    c2 = k2[0] - perpendicular_slope * k2[1]

    #solve two linear equations
    A = np.array([[slope, -1], [perpendicular_slope, -1]])
    B = np.array([-c1, -c2])
    new_origin = np.linalg.solve(A, B)

    angle = np.arctan(perpendicular_slope)
    angle_degree = np.degrees(angle)

    M = cv2.getRotationMatrix2D((new_origin[1], new_origin[0]), angle_degree, 1.0)
    return M
def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    original_image = img
    # TODO threshold and blur
    binarized_smooth_image = binarizeAndSmooth(img)
    # plt.imshow(binarized_smooth_image, cmap='gray')
    # plt.show()

    # TODO find and draw largest contour in image
    largest_contour_img = drawLargestContour(binarized_smooth_image)
    # plt.imshow(largest_contour_img, cmap='gray')
    # plt.show()

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    x_1 = 10
    x_2 = 20
    y_1 = getFingerContourIntersections(largest_contour_img, x_1)
    y_2 = getFingerContourIntersections(largest_contour_img, x_2)

    # plt.imshow(largest_contour_img, cmap='gray')
    # plt.axvline(x=x_1, color='r', linestyle='--', linewidth=1)
    # plt.axvline(x=x_2, color='r', linestyle='--', linewidth=1)

    # TODO compute middle points from these contour intersections
    m_1_x_1 = (x_1, int((y_1[0] + y_1[1]) / 2))
    m_2_x_1 = (x_1, int((y_1[2] + y_1[3]) / 2))
    m_3_x_1 = (x_1, int((y_1[4] + y_1[5]) / 2))

    m_1_x_2 = (x_2, int((y_2[0] + y_2[1]) / 2))
    m_2_x_2 = (x_2, int((y_2[2] + y_2[3]) / 2))
    m_3_x_2 = (x_2, int((y_2[4] + y_2[5]) / 2))

    # plt.scatter(m_1_x_1[0], m_1_x_1[1], color='blue', s=10)
    # plt.scatter(m_2_x_1[0], m_2_x_1[1], color='blue', s=10)
    # plt.scatter(m_3_x_1[0], m_3_x_1[1], color='blue', s=10)
    #
    # plt.scatter(m_1_x_2[0], m_1_x_2[1], color='green', s=10)
    # plt.scatter(m_2_x_2[0], m_2_x_2[1], color='green', s=10)
    # plt.scatter(m_3_x_2[0], m_3_x_2[1], color='green', s=10)

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(largest_contour_img, m_1_x_1[1], m_1_x_1[0], m_1_x_2[1], m_1_x_2[0])
    k2 = findKPoints(largest_contour_img, m_2_x_1[1], m_2_x_1[0], m_2_x_2[1], m_2_x_2[0])
    k3 = findKPoints(largest_contour_img, m_3_x_1[1], m_3_x_1[0], m_3_x_2[1], m_3_x_2[0])

    # plt.scatter(k1[1], k1[0], color='brown', s=10)
    # plt.scatter(k2[1], k2[0], color='brown', s=10)
    # plt.scatter(k3[1], k3[0], color='brown', s=10)
    # plt.show()

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    rotation_matrix = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    height, width = original_image.shape
    rotated_image = cv2.warpAffine(src=original_image, M=rotation_matrix, dsize=(width, height))
    # plt.imshow(rotated_image, cmap='gray')
    # plt.show()

    return rotated_image

# img1 = cv2.imread('Hand2.jpg', cv2.IMREAD_GRAYSCALE)
#
# palmPrintAlignment(img1)
