"""Just to test some stuff.
right now I am testing StereoUtils from the Common repository
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

from Common import StereoUtils, Decorators

# Adding run time prints to some of the functions
StereoUtils.read_catalogue = Decorators.running_time(StereoUtils.read_catalogue)
StereoUtils.l0_l1_conversion = Decorators.running_time(StereoUtils.l0_l1_conversion)
StereoUtils.image_preprocessing = Decorators.running_time(StereoUtils.image_preprocessing)

# Getting the catalogue
catalogue_df = StereoUtils.read_catalogue()

# Testing on some filenames
filenames = catalogue_df['filename'][:200:15]
# full_paths = StereoUtils.fullpath(filenames)

l1_images = StereoUtils.l0_l1_conversion(filenames)
# l1_images = StereoUtils.l0_l1_conversion(full_paths)

preprocessed_images = StereoUtils.image_preprocessing(l1_images, clip_percentages=(1, 99.99), log=False)
print('preprocessed images done')

