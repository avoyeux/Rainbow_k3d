"""
Small function to change the original masks to grayscale uint8 masks with white (i.e. 255) being the null values.
"""

# Imports
import os
import numpy as np
from PIL import Image

from pathlib import Path


mask_paths = Path("../STEREO/masque_karine_notprocessed").glob('*.png')
nw_path = "../STEREO/masque_karine"
os.makedirs(nw_path, exist_ok=True)

for mask_path in mask_paths:
    image = Image.open(mask_path)
    image = np.array(image)
    image = np.mean(image, axis=2)

    image = (~(image > 0) * 255).astype('uint8')
    image = np.stack((image, image, image), axis=-1)
    image = Image.fromarray(image)
    image.save(os.path.join(nw_path, 'frame' + os.path.basename(mask_path)[:4] + '.png'))


