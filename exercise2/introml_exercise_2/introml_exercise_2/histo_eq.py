# Implement the histogram equalization in this file
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# (a) Load image
image = Image.open('hello.png')

# Convert to grayscale
image = image.convert('L')

# Convert to numpy array
img_array = np.array(image)

# (b) Compute the intensity histogram
histogram = np.zeros(256, dtype=int)
for pixel_value in img_array.flatten():
    histogram[pixel_value] += 1

# Verify histogram
print(f"Sum of the first 90 values of the histogram: {np.sum(histogram[:90])}")

# (c) Compute its cumulative distribution function C
cdf = np.zeros(256, dtype=float)
cdf[0] = histogram[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + histogram[i]


# Normalize CDF
cdf = cdf / cdf[-1]


plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(cdf)

plt.subplot(2, 2, 2)
plt.plot(histogram)

# Verify CDF
print(f"Sum of the first 90 values of the CDF: {np.sum(cdf[:90])}")

# (d) Change the gray value of each pixel
new_img_array = np.zeros_like(img_array)
cdf_min = cdf[cdf > 0].min()

for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        pixel_value_old = img_array[i, j]
        pixel_value_new = int((cdf[pixel_value_old] - cdf_min) / (1 - cdf_min) * 255)
        new_img_array[i, j] = pixel_value_new


histogram = np.zeros(256, dtype=int)
for pixel_value in new_img_array.flatten():
    histogram[pixel_value] += 1

cdf = np.zeros(256, dtype=float)
cdf[0] = histogram[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + histogram[i]
cdf = cdf / cdf[-1]
plt.subplot(2, 2, 3)
plt.plot(cdf)

plt.subplot(2, 2, 4)
plt.plot(histogram)
plt.show()

# Convert the new image array back to an image
new_image = Image.fromarray(new_img_array)

# Save the result
new_image.save('kitty.png')
