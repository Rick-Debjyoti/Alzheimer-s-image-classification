
# ## SEGMENTATION METHODS 
# 
# - ### `Multi-Otsu Thresholding` 
#     - The multi-Otsu threshold is a thresholding algorithm that is used to separate the pixels of an input image into several different classes, each one obtained according to the intensity of the gray levels within the image.
#        Multi-Otsu calculates several thresholds, determined by the number of desired classes. The default number of classes is 3, the algorithm returns two threshold values. They are represented by a red line in the histogram below.
#
# - ### `Region-based Segmentation`
#     -  The region-based segmentation employs the Sobel filter to generate an elevation map from the input image. Markers are strategically placed, and the Watershed algorithm is applied to segment the image. 
#        Subsequently, holes in the regions are filled, and connected components are labeled for a comprehensive segmentation result.



# ## Importing the required libraries
import os
import numpy as np 
import cv2 as cv
import pandas as pd 
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import threshold_multiotsu
from skimage.filters import sobel
from skimage import segmentation as segm
from scipy import ndimage as ndi
from skimage.color import label2rgb


# ## Loading the images
img1 = io.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/MildDemented/mildDem0.jpg")
img2 = io.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/ModerateDemented/moderateDem0.jpg")
img3 = io.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/NonDemented/nonDem0.jpg")
img4 = io.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/VeryMildDemented/verymildDem0.jpg")
images = [img1, img2, img3, img4]


## Creating a class for segmentation
class Segmentation:
    """(list) -> None
    This class takes a list of images and segments them using the Multi-Otsu thresholding and region based watershed segmentation methods.
    Args:
        images (list): A list of images to be processed.
    """
    def __init__(self, images):
        self.images = images
        self.segmented_images_otsu = []
        self.thresholds_otsu = []
        self.regions_otsu = []
        self.segmented_images_region = []
        self.elevation_map_regions = []
        self.markers_regions = []
        self.segmentation_regions = []
        self.image_label_overlay_regions = []

    def otsu_segment_images(self , no_of_classes=3):
        """ (int) -> None
        This method takes the list of images and segments them using the Multi-Otsu thresholding method. takes the no of classes as an argument."""
        for image in self.images:
            thresholds = threshold_multiotsu(image, classes=no_of_classes)
            regions = np.digitize(image, bins=thresholds)
            self.thresholds_otsu.append(thresholds)
            self.regions_otsu.append(regions)
            self.segmented_images_otsu.append(regions)

    def plot_otsu_segmentation(self):
        """(None) -> None
        This method plots the original images with the histogram and the segmented images using multi-otsu methods with labels."""
        fig, axes = plt.subplots(len(self.images), 3, figsize=(15, 20))
        for i in range(3):
            for j in range(len(self.images)):
                if i == 0:
                    axes[j, i].imshow(self.images[j], cmap='gray') 
                    axes[j, i].set_title(f'Original-Image {j+1}')
                elif i == 1:
                    axes[j, i].hist(self.images[j].ravel(), bins=255, range=(5, 255))
                    axes[j, i].set_title(f'Histogram-Image {j+1}')
                    for thresh in self.thresholds_otsu[j]:
                        axes[j, i].axvline(thresh, color='r')
                elif i == 2:
                    axes[j, i].imshow(self.segmented_images_otsu[j], cmap='gray')
                    axes[j, i].set_title(f'Segmented-Image {j+1}')
                axes[j, i].axis('on')
        plt.subplots_adjust()
        plt.show()
    


    def region_based_segmentation(self):
        """ (None) -> None
        This method takes the list of images and segments them using the region based watershed segmentation method."""
        for image in self.images:
            elevation_map = sobel(image)

            markers = np.zeros_like(image)
            markers[image < 30] = 1
            markers[image > 150] = 2

            segmentation_region_method = segm.watershed(elevation_map, markers)

            segmentation_region = ndi.binary_fill_holes(segmentation_region_method - 1)

            labeled_image, _ = ndi.label(segmentation_region)
            image_label_overlay = label2rgb(labeled_image, image=image)

            self.elevation_map_regions.append(elevation_map)
            self.markers_regions.append(markers)
            self.segmentation_regions.append(segmentation_region)
            self.image_label_overlay_regions.append(image_label_overlay)


    def plot_region_segmentation(self):   
        """(None) -> None
        This method plots the original images with the contour and the segmented images using region based methods with labels."""
        fig, axes = plt.subplots(len(self.images), 2, figsize=(15,20), sharex=True, sharey=True)
        for i in range(len(self.images)):
            axes[i, 0].imshow(images[i], cmap=plt.cm.gray, interpolation='nearest')
            axes[i, 0].contour(self.segmentation_regions[i], [0.5], linewidths=1.2, colors='y')
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'Original Image {i+1} with Contour')

            axes[i, 1].imshow(self.image_label_overlay_regions[i], interpolation='nearest')
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'Segmentation Result {i+1} with Labels')
        plt.show()

    



# Example of using the class
segmentation = Segmentation(images)
segmentation.otsu_segment_images(no_of_classes=4) #otsu segmentation
segmentation.plot_otsu_segmentation()


segmentation.region_based_segmentation() #region based segmentation
segmentation.plot_region_segmentation()