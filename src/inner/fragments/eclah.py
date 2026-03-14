import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_eclah(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

image = cv2.imread("mammography_image.png")
if image is None: exit()

eclah_image = apply_eclah(image)

plt.figure(figsize=(10, 5))
for idx, (img, title) in enumerate([(image, "Original Image"), (eclah_image, "ECLAH Enhanced Image")], 1):
    plt.subplot(1, 2, idx)
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.show()