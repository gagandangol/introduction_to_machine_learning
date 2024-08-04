import numpy as np
from matplotlib import pyplot as plt


def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    # TODO

    if duration == 0:
        raise ValueError("Duration cannot be 0.")

    if freqfrom == 0:
        raise ValueError("Frequency cannot be 0.")

    t = np.linspace(0, duration, (samplingrate * duration))

    if linear:
        k = (freqto - freqfrom) / duration
        signal = np.sin(2 * np.pi * (freqfrom * t + 0.5 * k * t ** 2))
    else:
        k = (freqto / freqfrom) ** 1 / duration
        signal = np.sin(2 * np.pi * freqfrom * (k ** t - 1) / np.log(k))

    # plt.plot(t, signal)
    # plt.show()
    return signal

# createChirpSignal(200, 1, 1, 10, False)