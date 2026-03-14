import cv2
import numpy as np
import matplotlib.pyplot as plt

def noise_reduction_mscf(image, kernel_size=5, sigma=1):
    gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    median = cv2.medianBlur(image, kernel_size)
    return gaussian if np.var(gaussian) < np.var(median) else median

image = cv2.imread("mammography_image.png", cv2.IMREAD_GRAYSCALE)
if image is None: exit()

filtered_image = noise_reduction_mscf(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Filtered Image (MSCF)")
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.show()