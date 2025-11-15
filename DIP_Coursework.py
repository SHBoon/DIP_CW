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
from scipy import ndimage
import time

def load_image(image_path):
   """Load an image from the specified path."""
   image = plt.imread(image_path)
   return image

def basic_median_filter(image, mask_size=3):
   """
   Apply a median filter to an image. 
   Ineficient implementation as median is 
   computed from scratch for each pixel.
   """
   start_time = time.time()

   # Pad the image to handle borders
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   median_filtered_image = np.zeros_like(image)

   # Cycle through each pixel in the image
   for i in range(image.shape[0]):
      x = i + (mask_size - 1) // 2
      for j in range(image.shape[1]):
         y = j + (mask_size - 1) // 2
         # Extract the neighborhood
         neighborhood = padded_image[(x-(mask_size-1)//2):(x+(mask_size-1)//2), (y-(mask_size-1)//2):(y+(mask_size-1)//2)]

         # Compute the median and assign to the output image
         median_filtered_image[i, j] = np.median(neighborhood)

   end_time = time.time()
   print(f"Basic median filter applied in {end_time - start_time:.2f} seconds.")

   return median_filtered_image

def Huang_median_filter(image, mask_size=3):
   """
   Apply an efficient median filter to an image using Huang's algorithm.
   """
   start_time = time.time()
   
   # Pad the image to handle borders
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   median_filtered_image = np.zeros_like(image)

   # Cycle through each pixel in the image
   for i in range(image.shape[0]):
      x = i + (mask_size - 1) // 2
      for j in range(image.shape[1]):
         y = j + (mask_size - 1) // 2
         # Extract the neighborhood
         
   
   end_time = time.time()
   print(f"Efficient median filter applied in {end_time - start_time:.2f} seconds.")
   return median_filtered_image

def basic_mean_filter(image, mask_size=3):
   """Apply a mean filter to an image."""

   time_start = time.time()

   coeff = 1 / (mask_size * mask_size)

   # Pad the image to handle borders
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   mean_filtered_image = np.zeros_like(image)
   
   # Cycle through each pixel in the image
   for i in range(image.shape[0]):
      x = i + (mask_size - 1) // 2
      for j in range(image.shape[1]):
         y = j + (mask_size - 1) // 2
         # Extract the neighborhood
         neighborhood = padded_image[(x-(mask_size-1)//2):(x+(mask_size-1)//2), (y-(mask_size-1)//2):(y+(mask_size-1)//2)]

         mean_filtered_image[i, j] = coeff * np.sum(neighborhood)

   time_end = time.time()
   print(f"Basic mean filter applied in {time_end - time_start:.2f} seconds.")

   return mean_filtered_image

def efficient_mean_filter(image, mask_size=3):
   """Apply an efficient mean filter to an image."""
   
   # TBC
   
   pass

def sharpening_filter(image):
   """Apply a sharpening filter to an image."""

   time_start = time.time()
   
   # Sharpening Kernel
   K8 = np.array([[ -1, -1, -1],
                  [ -1,  8, -1],
                  [ -1, -1, -1]])
   
   K5 = np.array([[ 0, -1,  0],
                  [-1,  5, -1],
                  [ 0, -1,  0]])

   # Convolve with sharpening kernel
   sharpened_image = ndimage.convolve(image.astype(float), K5)
   
   # Clip values to valid range
   sharpened_image = np.clip(sharpened_image, 0, 255)

   time_end = time.time()
   print(f"Sharpening filter applied in {time_end - time_start:.2f} seconds.")

   return sharpened_image

def edge_detection(image, method='sobel3'):
   """
   Apply edge detection to an image to evaluate 
   filter performance. Uses the specified method.
   """
   
   if method == 'sobel3':
   # Sobel Kernels
      Kx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

      Ky = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

      # Convolve with Sobel kernels
      Ix = ndimage.convolve(image.astype(float), Kx)
      Iy = ndimage.convolve(image.astype(float), Ky)

      # Edge magnitude
      G = np.hypot(Ix, Iy)       # sqrt(Ix^2 + Iy^2)
      G = G / G.max() * 255      # Normalize to 0–255
   
   return G

if __name__ == "__main__":
   # Load image
   image_path = "Files/NZjers1.png"
   pre_process_image = np.asarray(load_image(image_path))

   # Image processing
   post_process_image = basic_mean_filter(pre_process_image,5)
   post_process_image = basic_median_filter(post_process_image,5)
   post_process_image = sharpening_filter(post_process_image)

   # Edge detection for evaluation
   edges_before = edge_detection(pre_process_image)
   edges_after = edge_detection(post_process_image)

   # Display results
   plt.figure(figsize=(10, 5))

   # Pre-processed image
   plt.subplot(2, 2, 1)
   plt.imshow(pre_process_image)
   plt.title("Original Image")
   plt.axis('off')

   # Post-processed image
   plt.subplot(2, 2, 2)
   plt.imshow(post_process_image)
   plt.title("Processed Image")
   plt.axis('off')

   # Edges before processing
   plt.subplot(2, 2, 3)
   plt.imshow(edges_before, cmap='gray')
   plt.title("Edges Before Processing")
   plt.axis('off')

   # Edges after processing
   plt.subplot(2, 2, 4)
   plt.imshow(edges_after, cmap='gray')
   plt.title("Edges After Processing")
   plt.axis('off')

   # Display the figure
   plt.tight_layout()
   plt.show()