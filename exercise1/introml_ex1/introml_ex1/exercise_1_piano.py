
import numpy as np
import matplotlib.pyplot as plt

def load_sample(filename, duration=4*44100, offset=44100//10):
    file_loaded = np.load(filename)
    # plt.plot(file_loaded)
    # plt.show()
    max_index = np.argmax(file_loaded)
    start_index = max_index + offset
    end_index = start_index + duration
    signal_part = file_loaded[start_index:end_index]
    # plt.plot(signal_part)
    # plt.show()
    return signal_part

def compute_frequency(sample, min_freq=50):
    sample_fft = np.abs(np.fft.fft(sample))
    # print(sample_fft, 'fft')
    sample_rate = 44100

    frequencies = np.fft.fftfreq(len(sample), 1 / sample_rate) #len(sample) is number of points and 1 / sample_rate
    # print(frequencies)

    #setting the frequencies smaller than min_frequencies to zero
    frequencies[frequencies < min_freq] = 0

    # finding indexes of required frequencies
    required_index = np.where(frequencies > 0)

    # finding the required frequencies
    frequencies = frequencies[required_index]

    # finding the pick index value of sample
    peak_index = np.argmax(sample_fft[required_index])

    # finding the main frequency
    main_frequency = frequencies[peak_index]
    return main_frequency

if __name__ == '__main__':

    file_names = ['sounds/Piano.ff.A2.npy', 'sounds/Piano.ff.A3.npy', 'sounds/Piano.ff.A4.npy',
                  'sounds/Piano.ff.A5.npy', 'sounds/Piano.ff.A6.npy', 'sounds/Piano.ff.A7.npy',
                  'sounds/Piano.ff.XX.npy']

    for file_name in file_names:
        sample = load_sample(file_name)
        frequency = compute_frequency(sample)
        print(f"File: {file_name}, Frequency: {frequency}")
        # plt.plot(sample)
        # plt.title(f"Sample from {file_name}")
        # plt.show()

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies

# 'A2' = 110.0,
# 'A3': 220.0,
# 'A4': 440.0,
# 'A5': 880.0,
# 'A6': 1760.0,
# 'A7': 3520.0,
# mistery frequency = 1179.8780167508098 which is closest to D6
