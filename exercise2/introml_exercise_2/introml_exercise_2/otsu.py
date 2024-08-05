import numpy as np
from matplotlib import pyplot as plt

#
# NO OTHER IMPORTS ALLOWED
#

import cv2

def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    # Initialize the histogram array with zeros for each possible pixel value (0-255)
    histogram = np.zeros(256, dtype=int)

    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Flatten the image to make it easier to convert 2-D array to 1-D array.
    flattened_img = img.flatten()

    # Count the occurrences of each pixel value
    for pixel_value in flattened_img:
        histogram[pixel_value] += 1

    # plt.plot(histogram)
    # plt.show()

    return histogram

# create_greyscale_histogram('kitty.png')

def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    binarized_img = np.where(img > t, 255, 0).astype(np.uint8)
    return binarized_img


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    total = np.sum(hist)

    # Compute p0 and p1
    p0 = np.sum(hist[:theta + 1]) / total
    p1 = np.sum(hist[theta + 1:]) / total

    return p0, p1

def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    sum0 = 0
    sum1 = 0

    # Compute sum0 for mu0
    for i in range(0, theta + 1):
        sum0 += i * hist[i]

    # Compute sum1 for mu1
    for i in range(theta + 1, len(hist)):
        sum1 += i * hist[i]

    # Calculate mu0 and mu1
    mu0 = sum0 / p0 if p0 != 0 else 0
    mu1 = sum1 / p1 if p1 != 0 else 0

    return mu0, mu1
def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''
    # TODO initialize all needed variables

    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    total = np.sum(hist)

    prob_hist = hist / total  # Change the histogram to visualize the probability distribution
    max_variance = 0
    best_theta = 0

    # TODO loop through all possible thetas
    for theta in range(1, len(hist) - 1):

        # TODO compute p0 and p1 using the helper function
        p0, p1 = p_helper(prob_hist, theta)

        # TODO compute mu and m1 using the helper function
        mu0, mu1 = mu_helper(prob_hist, theta, p0, p1)

        # TODO compute variance
        variance = p0 * p1 * (mu0 - mu1) ** 2

        # TODO update the threshold
        if variance > max_variance:
            max_variance = variance
            best_theta = theta
    return best_theta


def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    # TODO
    # Calculate histogram
    hist = create_greyscale_histogram(img)

    # Calculate Otsu threshold
    threshold = calculate_otsu_threshold(hist)

    # Binarize image using the threshold
    binarized_img = binarize_threshold(img, threshold)
    cv2.imshow('test', binarized_img)
    cv2.waitKey(0)
    return binarized_img

otsu(cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE))
