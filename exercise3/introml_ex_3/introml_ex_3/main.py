import cv2

from CannyEdgeDetector import canny

if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    canny(img)