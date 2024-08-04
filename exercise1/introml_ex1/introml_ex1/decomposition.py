import numpy as np
import matplotlib.pyplot as plt


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)

    # signal = []

    for k in range(0, k_max):
        signal += (8 / (np.pi ** 2)) * ((-1)**k) * (np.sin(2 * np.pi * (2 * k + 1) * frequency * t) / (2*k+1)**2)

    # plt.plot(t, signal)
    # plt.show()
    return signal

# createTriangleSignal(200, 2, 10)

def createSquareSignal(samples: int, frequency: int, k_max: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)
    for k in range(1, k_max):
        signal += (4 / np.pi) * (np.sin(2 * np.pi * frequency * (2*k-1) * t) / (2*k-1))
        # plt.plot(t, signal)
        # plt.show()
    return signal

# createSquareSignal(100, 2, 100)
def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    # returns the signal as 1D-array (np.ndarray)
    # TODO
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)
    signal = signal + amplitude/2
    for k in range(1, k_max + 1):
        signal -= (amplitude / np.pi) * (np.sin(2 * np.pi * frequency * k * t) / k)

    return signal

