# Title: DIP_Coursework.py
# Author: Sol Boon
# Date: 07/11/2025
# Version: 0.1

# Description: 
#   This script is part of a coursework assignment for a Digital Image Processing (DIP) course.
#   It includes functions for image processing tasks such as loading images, applying filters,
#   and saving processed images. The script is structured to facilitate easy extension and modification
#   for various image processing techniques.

import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
   """Load an image from the specified path."""
   image = plt.imread(image_path)
   return image

def basic_median_filter(image, mask_size=3):
   """Apply a median filter to an image."""
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   median_filtered_image = np.zeros_like(image)

   for i in range(image.shape[0]):
       for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+mask_size, j:j+mask_size]
            median_filtered_image[i, j] = np.median(neighborhood)
   
   return median_filtered_image

def mean_filter(image, mask_size=3):
   """Apply a mean filter to an image."""
   
   # TBC
   
   pass

if __name__ == "__main__":
   # Load image
   image_path = "Files/NZjers1.png"
   pre_process_image = np.asarray(load_image(image_path))

   # Image processing
   post_process_image = basic_median_filter(pre_process_image)

   # Display results
   plt.figure(figsize=(10, 5))

   # Pre-processed image
   plt.subplot(1, 2, 1)
   plt.imshow(pre_process_image)
   plt.title("Original Image")
   plt.axis('off')

   # Post-processed image
   plt.subplot(1, 2, 2)
   plt.imshow(post_process_image)
   plt.title("Processed Image")
   plt.axis('off')

   # Display the figure
   plt.tight_layout()
   plt.show()