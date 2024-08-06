from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    sum_val = 0.0
    for x in range(ksize):
        for y in range(ksize):
            dist = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-dist / (2 * sigma ** 2))
            sum_val += kernel[x, y]

    kernel /= sum_val  # Normalize the kernel
    return kernel


def slow_convolve(arr, k):
    k = np.flipud(np.fliplr(k))  # Flip the kernel both horizontally and vertically
    k_height, k_width = k.shape
    arr_height, arr_width = arr.shape

    pad_height = k_height // 2
    pad_width = k_width // 2

    # Padding the input image
    padded_arr = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    new_arr = np.zeros((arr_height, arr_width), dtype=np.float32)

    # Performing the convolution
    for i in range(arr_height):
        for j in range(arr_width):
            region = padded_arr[i:i + k_height, j:j + k_width]
            new_arr[i, j] = np.sum(region * k)

    return new_arr


if __name__ == '__main__':
    k = make_kernel(3, 1)  # Parameters for the Gaussian kernel, you can adjust sigma

    # Choose the image you prefer
    # TODO: chose the image you prefer
    # im = np.array(Image.open('input1.jpg'))
    im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    #TODO: blur  the image, subtract the resulttothe input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result

    sharpened_im = np.zeros_like(im)
    for i in range(3):  # Apply unsharp masking to each channel
        blurred_channel = slow_convolve(im[:, :, i], k)
        mask_channel = im[:, :, i] - blurred_channel
        sharpened_im[:, :, i] = np.clip(im[:, :, i] + mask_channel, 0, 255)

    sharpened_im = sharpened_im.astype(np.uint8)

    result_image = Image.fromarray(sharpened_im)
    result_image.save('sharpened_image_2.jpg')
