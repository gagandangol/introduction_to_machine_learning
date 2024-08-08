import numpy as np
import matplotlib.pyplot as plt

def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    height, width = shape

    # Convert polar to Cartesian coordinates
    x = int(r * np.cos(theta) + width / 2)
    y = int(r * np.sin(theta) + height / 2)

    return y, x

def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    Return Magnitude in Decibel
    :param img:
    :return:
    '''
    f = np.fft.fft2(img)
    # plt.plot(f)
    # plt.show()

    fshift = np.fft.fftshift(f)
    # plt.plot(fshift)
    # plt.show()

    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum + 0.000000001)

    # plt.plot(magnitude_spectrum)
    # plt.show()

    return magnitude_spectrum_db

def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring --> theta/sampling rate
    :return: feature vector of k features
    '''

    rings = np.zeros(k)
    height, width = magnitude_spectrum.shape

    max_radius = min(height, width)

    for i in range(1, k + 1):
        for j in range(sampling_steps):
            theta = np.pi * j / (sampling_steps - 1)
            for r in range(k * (i - 1), k * i + 1):
                y, x = polarToKart((height, width), r, theta)
                if y < max_radius and x < max_radius:
                    rings[i - 1] += magnitude_spectrum[y, x]
                else:
                    break
    return rings

def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area --> theta/sampling rate
    :return: feature vector of length k
    """
    fans = np.zeros(k)

    height, width = magnitude_spectrum.shape

    max_radius = min(height // 2, width // 2)
    for i in range(1, k + 1):
        for j in range(sampling_steps):
            theta = (i * np.pi / k) * j / (sampling_steps - 1)
            for r in range(max_radius):
                y, x = polarToKart((height, width), r, theta)
                fans[i - 1] += magnitude_spectrum[y, x]

    return fans

def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    plt.imshow(magnitude_spectrum)
    plt.show()
    ring_feature = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    fan_feature = extractFanFeatures(magnitude_spectrum, k, sampling_steps)

    return ring_feature, fan_feature