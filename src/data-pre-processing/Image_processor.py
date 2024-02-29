
# ## INTENSITY NORMALIZATION METHODS 
# 
# - ### `Histogram equilization` and `Contrast Limiting Adaptive Histogram Equilization (CLAHE)` 
#     - Histogram equilization is the method to strech the pixel intensity histogram of an image to the full range. That is for an 8-bit image the range is 0 to 255.
#        This is done for a balanced constrast in the images. Often we see this type of streching either gives too bright or too dark regions in the images, 
#        thus to tackle this problem we have Contrast limiting Adaptive methods like CLAHE , which does histogram equiulization in local contrast ranges by choosing a smaller grid (in our case (8x8)). 
#        This method prevents the image from washing out and helps preserve some features in the images.
## - ### `Z-score normalization`
#        - Z-score normalization is a method to normalize the pixel intensities of an image by subtracting the mean and dividing by the standard deviation of the image. 
#          This method is used to normalize the pixel intensities of an image to a standard normal distribution.
#
# - ### `Zero-one normalization`
#        - Zero-one normalization is a method to normalize the pixel intensities of an image by subtracting the minimum pixel intensity and dividing by the difference between the maximum and minimum pixel intensities of the image. 
#          This method is used to normalize the pixel intensities of an image to the range 0 to 1.
# - ### `Percentile normalization`
#        - Percentile normalization is a method to normalize the pixel intensities of an image by subtracting the nth percentile and dividing by the difference between the mth and nth percentile pixel intensities of the image.
#           This method is used to normalize the pixel intensities of an image to the range 0 to 1.
# - ### `Resizing`
#        - Resizing is a method to resize the image to a given target size. This method is used to resize the images to a standard size.


# ## IMPORTS
import os
import numpy as np 
import cv2 as cv
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image


## 1 image of each class from kaggle data
img1 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/MildDemented/mildDem0.jpg")
img2 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/ModerateDemented/moderateDem0.jpg")
img3 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/NonDemented/nonDem0.jpg")
img4 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/VeryMildDemented/verymildDem0.jpg")
images = [img1, img2, img3, img4]



# Class for image pre-processing
class ImageProcessor:
    """ (list,float,tuple)->None    
    Takes a list of images and performs image processing using histogram equalization and CLAHE methods and also performs image resizing, z-score, zero-one and percentile normalization.
    Args:
        images (list): A list of images to be processed.
        clipLimit (float, optional): The clip limit parameter for CLAHE. Defaults to 5.0.
        tileGridSize (tuple, optional): The tile grid size parameter for CLAHE. Defaults to (8, 8).
        
        """
    
    def __init__(self, images, CLAHE_clipLimit = 5.0, CLAHE_tileGridSize = (8,8), minP=0 ,maxP=99):
        self.images = images
        self.hist_eq_images = []
        self.CLAHE_images = []
        self.hist_eq_images_out = []
        self.CLAHE_images_out = []
        self.z_score_images = []
        self.zero_one_norm_images = []
        self.percentile_norm_images = []
        self.resized_images = []  
        self.clipLimit = CLAHE_clipLimit
        self.tileGridSize = CLAHE_tileGridSize
        self.minP = minP    
        self.maxP = maxP


    def hist_equ(self, img):
        lab_img = cv.cvtColor(img , cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab_img)
        equ = cv.equalizeHist(l)
        updated_lab_img = cv.merge((equ,a,b))
        hist_eq_img = cv.cvtColor(updated_lab_img , cv.COLOR_LAB2BGR)
        hist_eq_img_out = cv.cvtColor(hist_eq_img, cv.COLOR_BGR2GRAY)

        clahe = cv.createCLAHE(clipLimit = self.clipLimit, tileGridSize = self.tileGridSize)
        clahe_img = clahe.apply(l)
        updated_lab_img2 = cv.merge((clahe_img , a,b ))
        CLAHE_img = cv.cvtColor(updated_lab_img2 , cv.COLOR_LAB2BGR)
        CLAHE_img_out = cv.cvtColor(CLAHE_img, cv.COLOR_BGR2GRAY)
        return hist_eq_img, CLAHE_img, hist_eq_img_out, CLAHE_img_out
    

    def z_score_normalization(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mean, std = cv.meanStdDev(img)
        z_score_img = (img - mean) / std
        return z_score_img


    def zero_one_normalization(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv.minMaxLoc(img)
        zero_one_norm_img = (img - min_val) / (max_val - min_val)
        return zero_one_norm_img


    def percentile_normalization(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        minVal = np.nanpercentile(img, self.minP)
        maxVal = np.nanpercentile(img, self.maxP)
        percentile_norm_img = (img - minVal) * (1 / (maxVal - minVal))
        return percentile_norm_img


    def resize_images(self, target_size):
        for img in self.images:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            resized_img = cv.resize(gray, target_size)
            self.resized_images.append(resized_img)


    def process_images(self, method='hist_eq'):
        for i in range(len(self.images)):
            if method == 'hist_eq':
                hist_eq, clahe, hist_eq_out, CLAHE_out = self.hist_equ(self.images[i])
                self.hist_eq_images.append(hist_eq)
                self.CLAHE_images.append(clahe)
                self.hist_eq_images_out.append(hist_eq_out)
                self.CLAHE_images_out.append(CLAHE_out)
            elif method == 'z_score':
                z_score_img = self.z_score_normalization(self.images[i])
                self.z_score_images.append(z_score_img)
            elif method == 'zero_one_norm':
                zero_one_norm_img = self.zero_one_normalization(self.images[i])
                self.zero_one_norm_images.append(zero_one_norm_img)
            elif method == 'percentile_norm':
                percentile_norm_img = self.percentile_normalization(self.images[i])
                self.percentile_norm_images.append(percentile_norm_img)


    def plot_images_intensity_norm(self):
        fig, axes = plt.subplots(len(self.images), 3, figsize=(15, 20))
        for i in range(3):
            for j in range(len(self.images)):
                if i == 0:
                    axes[j, i].imshow(self.images[j], cmap='gray') 
                    axes[j, i].set_title(f'Original-Image {j+1}')
                elif i == 1:
                    axes[j, i].imshow(self.hist_eq_images[j], cmap='gray')
                    axes[j, i].set_title(f'Hist_eq-Image {j+1}')
                elif i == 2:
                    axes[j, i].imshow(self.CLAHE_images[j], cmap='gray')
                    axes[j, i].set_title(f'CLAHE- Image {j+1}')
                axes[j, i].axis('off')
                

    def plot_histograms(self):
        fig, axes = plt.subplots(len(self.images), 3, figsize=(15, 20))
        for i in range(3):
            for j in range(len(self.images)):
                if i == 0:
                    lab_img = cv.cvtColor(self.images[j] , cv.COLOR_BGR2LAB)
                    l, a, b = cv.split(lab_img)
                    axes[j, i].hist(l.flat , bins = 20)
                    axes[j, i].set_title(f'Original-Histogram {j+1}')
                elif i == 1:
                    lab_img = cv.cvtColor(self.hist_eq_images[j] , cv.COLOR_BGR2LAB)
                    l, a, b = cv.split(lab_img)
                    axes[j, i].hist(l.flat , bins = 20)
                    axes[j, i].set_title(f'Hist_eq-Histogram {j+1}')
                elif i == 2:
                    lab_img = cv.cvtColor(self.CLAHE_images[j] , cv.COLOR_BGR2LAB)
                    l, a, b = cv.split(lab_img)
                    axes[j, i].hist(l.flat , bins = 20)
                    axes[j, i].set_title(f'CLAHE-Histogram {j+1}')
                axes[j, i].axis('on')
        plt.tight_layout()
        plt.show()




## Trying the ImageProcessor class on 4 images of differnet classes of Kaggle data
processor = ImageProcessor(images, CLAHE_clipLimit=5.0, CLAHE_tileGridSize=(8,8))
processor.resize_images(target_size=(256, 256))  # processor.resized_images is the list of resized images
processor.process_images(method='z_score')    # processor.z_score_images is the list of z-score normalized images
processor.process_images(method='zero_one_norm')  # processor.zero_one_norm_images is the list of zero-one normalized images
processor.process_images(method='percentile_norm')  # processor.percentile_norm_images is the list of percentile normalized images
processor.process_images(method='hist_eq')  # processor.hist_eq_images_out is the list of histogram equalized images and processor.CLAHE_images_out is the list of CLAHE images
processor.plot_images_intensity_norm()
processor.plot_histograms() 



# function to convert the whole dataset to CLAHE or resize and preserve the folder structure (for other methods intensities b/w 0 and 1, hence not converted)
def preprocess_images_in_directory(input_directory, output_directory, processing_method='hist_eq_CLAHE', target_size=(128, 128)):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                output_path = input_path.replace(input_directory, output_directory)
                output_folder = os.path.dirname(output_path)

                os.makedirs(output_folder, exist_ok=True)
                image = cv.imread(input_path)

                processor = ImageProcessor([image])

                if processing_method == 'hist_eq_CLAHE':
                    processor.process_images(method='hist_eq')
                    processed_images = processor.CLAHE_images_out[0]
                elif processing_method == 'resize':
                    processor.resize_images(target_size=target_size)
                    processed_images = processor.resized_images[0]
                else:
                    raise ValueError(f"Unsupported processing method: {processing_method}")

                modified_image_pil = Image.fromarray(processed_images)
                modified_image_pil.save(output_path)




# Example
input_train_directory = "/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train"
output_train_directory = "/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/modified_train"

input_test_directory = "/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/test"
output_test_directory = "/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/modified_test"

preprocess_images_in_directory(input_train_directory, output_train_directory, processing_method='hist_eq_CLAHE')
preprocess_images_in_directory(input_test_directory, output_test_directory, processing_method='resize', target_size=(128, 128))