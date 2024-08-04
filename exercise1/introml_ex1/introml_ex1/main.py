import numpy as np
import matplotlib.pyplot as plt

from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal


# TODO: Test the functions imported in lines 1 and 2 of this file.


# Parameters
SAMPLING_RATE = 200
DURATION = 1
FREQ_FROM = 1
FREQ_TO = 10
SAMPLES = 200
FREQUENCY = 2
KMAX = 10000
AMPLITUDE = 1

if __name__ == '__main__':

    # Plot all signals in subplots
    plt.figure(figsize=(24, 20))

    # Plot linear chirp signal
    t = np.linspace(0, DURATION, (SAMPLING_RATE * DURATION))
    linear_chirp = createChirpSignal(SAMPLING_RATE, DURATION, FREQ_FROM, FREQ_TO, linear=True)
    plt.subplot(3, 2, 1)
    plt.plot(t, linear_chirp, color='blue')
    plt.title('Linear Chirp Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot exponential chirp signal
    exp_chirp = createChirpSignal(SAMPLING_RATE, DURATION, FREQ_FROM, FREQ_TO, linear=False)
    plt.subplot(3, 2, 2)
    plt.plot(t, exp_chirp, color='green')
    plt.title('Exponential Chirp Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot triangle signal
    t = np.linspace(0, DURATION, SAMPLES)
    triangle_signal = createTriangleSignal(SAMPLES, FREQUENCY, KMAX)
    plt.subplot(3, 2, 3)
    plt.plot(t, triangle_signal, color='orange')
    plt.title('Triangle Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot square signal
    plt.subplot(3, 2, 4)
    square_signal = createSquareSignal(SAMPLES, FREQUENCY, KMAX)
    plt.plot(t, square_signal, color='red')
    plt.title('Square Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot sawtooth signal
    plt.subplot(3, 2, 5)
    sawtooth_signal = createSawtoothSignal(SAMPLES, FREQUENCY, KMAX, AMPLITUDE)
    plt.plot(t, sawtooth_signal, color='purple')
    plt.title('Sawtooth Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.savefig('plots.png')
    plt.show()
