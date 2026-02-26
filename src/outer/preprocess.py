import numpy as np
from PIL import Image

def load_images(image_paths):
    processed_images = []

    for path in image_paths:
        image = process_image(path)
        processed_images.append(image)

    return np.vstack(processed_images)


def process_image(image_path):
    img = Image.open(image_path)

    return image


def normalize_image(image):
    image -= np.mean(image)
    image /= np.std(image)
