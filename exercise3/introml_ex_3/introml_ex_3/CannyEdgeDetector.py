import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from convo import make_kernel, slow_convolve

#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # TODO

    kernel = make_kernel(ksize, sigma)
    res = slow_convolve(img_in, kernel)
    return kernel, res.astype(int)

def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # TODO
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gx = slow_convolve(img_in, Kx)
    gy = slow_convolve(img_in, Ky)

    gx = gx.astype(int)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """

    # Calculate the gradient magnitude
    g = np.sqrt(gx ** 2 + gy ** 2)

    # Calculate the gradient direction
    theta = np.arctan2(gy, gx).astype(float)

    return g.astype(int), theta

def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    # TODO
    angle_deg = np.rad2deg(angle) % 180

    if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg < 180):
        return 0
    elif 22.5 <= angle_deg < 67.5:
        return 45
    elif 67.5 <= angle_deg < 112.5:
        return 90
    else:
        return 135

def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    M, N = g.shape
    Z = np.zeros((M, N), dtype=np.int32)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            angle = convertAngle(theta[i, j])
            q = 255
            r = 255

            if angle == 0:
                q = g[i, j + 1]
                r = g[i, j - 1]
            elif angle == 45:
                q = g[i + 1, j - 1]
                r = g[i - 1, j + 1]
            elif angle == 90:
                q = g[i + 1, j]
                r = g[i - 1, j]
            elif angle == 135:
                q = g[i - 1, j - 1]
                r = g[i + 1, j + 1]

            if (g[i, j] >= q) and (g[i, j] >= r):
                Z[i, j] = g[i, j]
            else:
                Z[i, j] = 0

    return Z

def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    # TODO
    thresh_img = np.zeros_like(max_sup)
    strong = 2
    weak = 1

    thresh_img[max_sup > t_high] = strong
    thresh_img[(max_sup > t_low) & (max_sup <= t_high)] = weak

    # Initialize the output image
    hysteresis_img = np.zeros_like(max_sup, dtype=np.uint8)

    rows, cols = max_sup.shape

    # Iterate over the image and process pixels
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if thresh_img[i, j] == strong:
                hysteresis_img[i, j] = 255
            elif thresh_img[i, j] == weak:
                if (thresh_img[i + 1, j - 1] == strong or thresh_img[i + 1, j] == strong or thresh_img[
                    i + 1, j + 1] == strong or
                        thresh_img[i, j - 1] == strong or thresh_img[i, j + 1] == strong or
                        thresh_img[i - 1, j - 1] == strong or thresh_img[i - 1, j] == strong or thresh_img[
                            i - 1, j + 1] == strong):
                    hysteresis_img[i, j] = 255

    return hysteresis_img

def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)
    plt.imshow(result, 'gray')
    plt.show()

    return result

