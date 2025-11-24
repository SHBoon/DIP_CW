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
      # Account for padding
      x = i + (mask_size - 1) // 2
      for j in range(image.shape[1]):
         # Account for padding
         y = j + (mask_size - 1) // 2

         # Extract the neighborhood
         neighborhood = padded_image[(x-(mask_size-1)//2):(x+(mask_size-1)//2), (y-(mask_size-1)//2):(y+(mask_size-1)//2)]

         # Compute the median and assign to the output image
         median_filtered_image[i, j] = neighborhood_median(neighborhood)

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

   # Initialize histogram
   hist = np.zeros(256)

   # Normalise
   padded_image = (padded_image * 255).astype(np.uint8)

   # Cycle through each pixel in the image
   for i in range(image.shape[0]):      
      # Account for padding
      x = i + (mask_size - 1) // 2

      hist[:] = 0

      # Initialize histogram for the first window
      for di in range(-(mask_size//2), (mask_size//2)+1):
         for dj in range(-(mask_size//2), (mask_size//2)+1):
            # Pixel values added to histogram
            pixel_value = padded_image[x + di, (mask_size//2) + dj]
            hist[pixel_value] += 1

      # Compute median for the first pixel in the row
      median_filtered_image[i, 0] = histogram_median(hist, mask_size)

      for j in range(1, image.shape[1]):
         # Account for padding
         y = j + (mask_size - 1) // 2
         
         # Update histogram: remove left column, add right column
         for di in range(-(mask_size//2), (mask_size//2)+1):

            # Pixel value leaving the window
            pixel_value_out = padded_image[x + di, y - (mask_size//2) - 1]
            hist[pixel_value_out] -= 1

            # Pixel value entering the window
            pixel_value_in = padded_image[x + di, y + (mask_size//2)]
            hist[pixel_value_in] += 1

         # Compute median for the current pixel
         median_filtered_image[i, j] = histogram_median(hist, mask_size)
   
   end_time = time.time()
   print(f"Huang efficient median filter applied in {end_time - start_time:.2f} seconds.")
   return median_filtered_image

def improved_Huang_median_filter(image, mask_size=3):
   """
   Apply an improved efficient median filter to an image using Huang's algorithm.
   """
   start_time = time.time()
   
   # Pad the image to handle borders
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   median_filtered_image = np.zeros_like(image)

   # Initialize histogram
   hist = np.zeros(256)

   # Normalise
   padded_image = (padded_image * 255).astype(np.uint8)

   # Cycle through each pixel in the image
   for i in range(image.shape[0]):      
      # Account for padding
      x = i + (mask_size - 1) // 2

      hist[:] = 0
      
      # In place of a nested loop, use slicing to initialize histogram for the first window
      w = mask_size // 2
      window = padded_image[x - w : x + w + 1, 0 : 2*w + 1]
      hist[:] = np.bincount(window.ravel(), minlength=256)

      # Compute median for the first pixel in the row
      median_filtered_image[i, 0] = histogram_median(hist, mask_size)

      for j in range(1, image.shape[1]):
         # Account for padding
         y = j + (mask_size - 1) // 2
         
         # Use slicing to get the columns going out and coming in
         column_out = padded_image[x - w : x + w + 1, y - w - 1]
         column_in = padded_image[x - w : x + w + 1, y + w]

         # Update histogram: remove left column, add right column
         hist -= np.bincount(column_out, minlength=256)
         hist += np.bincount(column_in, minlength=256)

         # Compute median for the current pixel
         median_filtered_image[i, j] = histogram_median(hist, mask_size)
   
   end_time = time.time()
   print(f"Improved Huang efficient median filter applied in {end_time - start_time:.2f} seconds.")
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

def sharpening_filter(image, type='K5'):
   """
   Apply a sharpening filter to an image.
   """

   time_start = time.time()
   
   # Sharpening Kernel
   K8 = np.array([[ -1, -1, -1],
                  [ -1,  8, -1],
                  [ -1, -1, -1]])
   
   K5 = np.array([[ 0, -1,  0],
                  [-1,  5, -1],
                  [ 0, -1,  0]])

   # Convolve with sharpening kernel
   if type == 'K8':
      sharpened_image = ndimage.convolve(image.astype(float), K8)
   elif type == 'K5':
      sharpened_image = ndimage.convolve(image.astype(float), K5)
   
   # Clip values to valid range
   sharpened_image = np.clip(sharpened_image, 0, 255)

   time_end = time.time()
   print(f"Sharpening filter applied in {time_end - time_start:.2f} seconds.")

   return sharpened_image

def unsharp_masking_filter(image, mask_size=3, k=1):
   """
   Apply an unsharp masking filter to an image.
   """

   # Pad the image to handle borders
   padded_image = np.pad(image, ((mask_size//2, mask_size//2), (mask_size//2, mask_size//2)), mode='edge')
   unsharp_filtered_image = np.zeros_like(image)
   
   # Cycle through each pixel in the image
   for i in range(image.shape[0]):
      x = i + (mask_size - 1) // 2
      for j in range(image.shape[1]):
         y = j + (mask_size - 1) // 2
         # Extract the neighborhood
         neighborhood = padded_image[(x-(mask_size-1)//2):(x+(mask_size-1)//2), (y-(mask_size-1)//2):(y+(mask_size-1)//2)]

         neighborhood_mean = np.mean(neighborhood)

         unsharp_filtered_image[i, j] = neighborhood_mean + k * (image[i, j] - neighborhood_mean)

   return unsharp_filtered_image

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
      G = np.hypot(Ix, Iy)       # Pythag
      G = G / G.max() * 255      # Normalize
   
   return G

def neighborhood_median(neighborhood):
   """
   Compute the median of a neighborhood.
   """
   
   list_values = []

   for i in range(neighborhood.shape[0]):
      for j in range(neighborhood.shape[1]):
         list_values.append(neighborhood[i, j])

   list_values.sort()

   median = list_values[len(list_values) // 2]

   return median

def histogram_median(hist, mask_size):
   median_pos = (mask_size * mask_size) // 2 + 1
   cdf = hist.cumsum()
   return np.searchsorted(cdf, median_pos)

if __name__ == "__main__":
   # Load image
   image_path = "Files/NZJers1.png"
   pre_process_image = np.asarray(load_image(image_path))

   # Image processing
   mask_size = 9

   post_process_image1 = basic_median_filter(pre_process_image, mask_size)
   post_process_image2 = Huang_median_filter(pre_process_image, mask_size)
   post_process_image3 = improved_Huang_median_filter(pre_process_image, mask_size)

   # Edge detection for evaluation
   edges_before = edge_detection(pre_process_image)
   edges_after1 = edge_detection(post_process_image1)
   edges_after2 = edge_detection(post_process_image2)
   edges_after3 = edge_detection(post_process_image3)

   # Display results
   plt.figure(figsize=(10, 5))

   # Pre-processed image
   plt.subplot(2, 4, 1)
   plt.imshow(pre_process_image)
   plt.title("Original Image")
   plt.axis('off')

   # Post-processed image
   plt.subplot(2, 4, 2)
   plt.imshow(post_process_image1)
   plt.title("Processed Image 1")
   plt.axis('off')

   # Post-processed image
   plt.subplot(2, 4, 3)
   plt.imshow(post_process_image2)
   plt.title("Processed Image 2")
   plt.axis('off')

   # Post-processed image
   plt.subplot(2, 4, 4)
   plt.imshow(post_process_image3)
   plt.title("Processed Image 3")
   plt.axis('off')

   # Edges before processing
   plt.subplot(2, 4, 5)
   plt.imshow(edges_before, cmap='gray')
   plt.title("Edges Before Processing")
   plt.axis('off')

   # Edges after processing
   plt.subplot(2, 4, 6)
   plt.imshow(edges_after1, cmap='gray')
   plt.title("Edges After Processing 1")
   plt.axis('off')

   # Edges after processing
   plt.subplot(2, 4, 7)
   plt.imshow(edges_after2, cmap='gray')
   plt.title("Edges After Processing 2")
   plt.axis('off')

   # Edges after processing
   plt.subplot(2, 4, 8)
   plt.imshow(edges_after3, cmap='gray')
   plt.title("Edges After Processing 3")
   plt.axis('off')

   # Display the figure
   plt.tight_layout()
   plt.show()