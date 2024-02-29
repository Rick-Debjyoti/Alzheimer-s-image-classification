
## FEATURE EXTRACTION METHODS 
## -Edge Detection:
##      -Roberts: Roberts: The Roberts edge detection method detects edges by highlighting areas with rapid changes in pixel intensity. 
##      -Sobel: The Sobel edge detection method detects edges by highlighting areas with high first-order derivatives in both horizontal and vertical directions.
##      -Scharr:  The Scharr edge detection method is similar to the Sobel method, but it provides better edge detection results, especially for images with high-frequency content.
##      -Prewitt: It is similar to the Sobel method but uses a slightly different kernel.
##      -Canny: The Canny edge detection method is a multi-stage algorithm that is widely used for edge detection. 
##               It involves noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding. 
##               The Canny method produces high-quality edge maps with well-defined edges and minimal noise. It produces a binary image.


## -Corner Detection:
##      -Harris Corner Detection: Harris corner detection algorithm is used to detect corners in an input image. This algorithm has three main steps.
##            1. Determine which part of the image has a large variation in intensity as corners have large variations in intensities. It does this by moving a sliding window throughout the image.
##            2. For each window identified, compute a score value R.
##            3. Apply threshold to the score and mark the corners.   
##      -Shi-Tomasi Corner Detection: This is another corner detection algorithm. It works similar to Harris Corner detection. 
##                                    The only difference here is the computation of the value of R. This algorithm also allows us to find the best n corners in an image.


## -Keypoint Detection:
##      -SIFT: The SIFT(Scale-Invariant Feature Transform) keypoint detection method is a keypoint detection method that is widely used in computer vision algorithms. SIFT is used to detect corners, blobs, circles, and so on. It is also used for scaling an image.      
##      -ORB: ORB(Oriented FAST and Rotated BRIEF) is a one-shot facial recognition algorithm. Here, two algorithms are involved. FAST and BRIEF. It works on keypoint matching. Key point matching of distinctive regions in an image like the intensity variations.
##





## Importing the required libraries
import os
import numpy as np 
import cv2 as cv
import pandas as pd 
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize,  downscale_local_mean
from skimage.filters import roberts, sobel, scharr , prewitt
from skimage.feature import canny



# Sample Kaggle images from each class of ALzheimer's Dataset 
img1 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/MildDemented/mildDem0.jpg")
img2 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/ModerateDemented/moderateDem0.jpg")
img3 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/NonDemented/nonDem0.jpg")
img4 = cv.imread("/Users/debjyoti_mukherjee/Downloads/Alzheimer_s Dataset/train/VeryMildDemented/verymildDem0.jpg")
images = [img1, img2, img3, img4]


## Scaling images to a standard size (example with img1)
rescaled_img1 = rescale(img1,0.25 ,anti_aliasing=True)  # anti-aliasing true to avoid fringes
plt.imshow(rescaled_img1, cmap='gray')
resized_img1 = resize(img1, (128,128), anti_aliasing=True)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
downscaled_img1 = downscale_local_mean(gray1, (4,3))   # downscale width by 4 and height by 3 , and does it along the mean.





# feature_extraction Class for images
class FeatureExtractor:
    """(list) -> None
        This class takes in a list of images and performs various feature extraction methods on the images.

        Args:
        images (list): A list of images to be processed.
    """

    def __init__(self, images):

        self.images = images
        self.roberts_images = []
        self.sobel_images = []
        self.scharr_images = []
        self.prewitt_images = []
        self.canny_images = []
        self.corner_harris_outputs = []
        self.shi_tomasi_outputs = []
        self.sift_outputs = []
        self.orb_outputs = []
        self.edge_outputs = [self.roberts_images, self.sobel_images, self.scharr_images, self.prewitt_images, self.canny_images]
        self.corner_outputs = [self.corner_harris_outputs, self.shi_tomasi_outputs] 
        self.keypoint_outputs = [self.sift_outputs, self.orb_outputs]
    
    def edge_detect(self, img, canny_sigma=1):
        """ (list, float) -> list
        This method takes in an image list and a canny_sigma value (for Canny-Edge detection) and performs various edge detection methods on the image."""
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edge_roberts = roberts(img_gray)
        edge_sobel = sobel(img_gray)
        edge_scharr = scharr(img_gray)
        edge_prewitt = prewitt(img_gray)
        edge_canny = canny(img_gray, sigma=canny_sigma)
        return edge_roberts, edge_sobel, edge_scharr, edge_prewitt, edge_canny
    

    def corner_detect(self, img):
        """(list) -> list
        This method takes in an image list and performs corner detection methods(Harris corner detection and Shi-Tomasi corner detection) on the image."""
        gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
        # Harris corner detection
        img_harris = img.copy()
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]
        # Shi-Tomasi corner detection
        corners = cv.goodFeaturesToTrack(gray, 20, 0.01, 10)
        corners = np.int0(corners) 
        img_shi_tomasi = img.copy()
        for i in corners:
            x, y = i.ravel()
            cv.circle(img_shi_tomasi, (x, y), 3, 255, -1)
        return img_harris, img_shi_tomasi
    

    def keypoint_detectors(self, img):
        """(list) -> list
        This method takes in an image list and performs keypoint detection(SIFT and ORB) methods on the image."""
        gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
        # SIFT keypoint detection
        sift = cv.SIFT_create()
        kp_sift, _ = sift.detectAndCompute(gray, None)
        img_sift = cv.drawKeypoints(gray, kp_sift, img.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # ORB keypoint detection
        orb = cv.ORB_create(nfeatures=200)
        kp_orb = orb.detect(img.copy(), None)
        kp_orb, des = orb.compute(img.copy(), kp_orb)
        img_orb = cv.drawKeypoints(img.copy(), kp_orb, None, color=(0, 255, 0), flags=0)
        return img_sift, img_orb
    
    
    
    
    def process_images(self, task_name, canny_sigma=1):
        """(list, str, float) -> None
        This method takes in an image list, a task_name and a canny_sigma value (for Canny-Edge detection) and performs various feature extraction methods on the images based on the task chosen."""
        for img in self.images:
            if task_name == "edge_detect":
                edge = self.edge_detect(img, canny_sigma)
                [self.edge_outputs[j].append(edge[j]) for j in range(len(self.edge_outputs))]
            elif task_name == "corner_detect":
                corner = self.corner_detect(img)
                [self.corner_outputs[j].append(corner[j]) for j in range(len(self.corner_outputs))]
            elif task_name == "keypoint_detect":
                key_des = self.keypoint_detectors(img)
                [self.keypoint_outputs[j].append(key_des[j]) for j in range(len(self.keypoint_outputs))]

    
    
    
    def plot_edge_images(self):
        """(None) -> None
        This method plots the original images and the edge detected images."""
        fig, axes = plt.subplots(len(self.images), 6, figsize=(20, 20))
        
        for i in range(6):
            for j in range(len(self.images)):
                if i == 0:
                    axes[j, i].imshow(self.images[j], cmap='gray') 
                    axes[j, i].set_title(f'Original-Image {j+1}')
                elif i == 1:
                    axes[j, i].imshow(self.edge_outputs[i-1][j], cmap='gray')
                    axes[j, i].set_title(f'Roberts {j+1}')
                elif i == 2:
                    axes[j, i].imshow(self.edge_outputs[i-1][j], cmap='gray')
                    axes[j, i].set_title(f'Sobel {j+1}')
                elif i == 3:
                    axes[j, i].imshow(self.edge_outputs[i-1][j], cmap='gray')
                    axes[j, i].set_title(f'Scharr {j+1}')
                elif i == 4:
                    axes[j, i].imshow(self.edge_outputs[i-1][j], cmap='gray')
                    axes[j, i].set_title(f'Prewitt {j+1}')
                elif i == 5:
                    axes[j, i].imshow(self.edge_outputs[i-1][j], cmap='gray')
                    axes[j, i].set_title(f'Canny {j+1}')
                axes[j, i].axis('off')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()







## Example Implementation
feature_extraction = FeatureExtractor(images)   # Create an instance of FeatureExtractor class

# Process images for edge detection
feature_extraction.process_images(task_name="edge_detect", canny_sigma=1)
feature_extraction.plot_edge_images()

# Process images for corner detection
feature_extraction.process_images(task_name="corner_detect")
plt.imshow(feature_extraction.corner_outputs[0][0], cmap='gray')  # Harris corner detection
plt.imshow(feature_extraction.corner_outputs[1][0], cmap='gray')  # Shi-Tomasi corner detection

# Process images for keypoint detection
feature_extraction.process_images(task_name="keypoint_detect")
plt.imshow(feature_extraction.keypoint_outputs[0][0], cmap='gray') # SIFT keypoint detection
plt.imshow(feature_extraction.keypoint_outputs[1][0], cmap='gray') # ORB keypoint detection

