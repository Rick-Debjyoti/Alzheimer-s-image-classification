# Alzheimer-s-image-classification
Analyzing Brain Scan Images for the Early Detection and Diagnosis of Alzheimer’s Disease

## Overview

This repository is a part of the Omdena Toronto Canada local chapter, `Analyzing Brain Scan Images for the Early Detection and Diagnosis of Alzheimer’s Disease` project lead by Kausthab Dutta Phukan. The project aims to leverage machine learning and computer vision techniques to analyze brain scan images for the early detection and diagnosis of Alzheimer's disease. In this repository, I have included the parts contributed by me in the pre-processing, modelling and presentation tasks. The Kaggle dataset with 6400 images across 4 different classes or stages of Alzheimer's diseasen is used for modelling and deployment after pre-processing and augmentation. Most of the pre-processing is done using `OpenCV` and `Scikit-Image` library, while `TensorFlow` is used for model building. The original project model is deployed on `Huggingface Spaces` with `Streamlit` app as front-end and a `docker` container as backend deployed as RESTful API. The detailed implementation of each section is mentioned in the presentation available in the `reports` folder.



## Preprocessing

The pre-processing steps inclues:

*   Intensity Normalisation
*   Segmentation
*   Feature Extraction

## Intensity normalisation 

*  **Histogram Equalization and CLAHE (Contrast Limiting Adaptive Histogram Equalization)**: Histogram equalization stretches the pixel intensity histogram of an image to the full range, enhancing contrast. CLAHE performs histogram equalization in local contrast ranges, preventing image washout and preserving features.
*  **Z-score normalization** subtracts the mean and divides by the standard deviation, standardizing pixel intensities to a normal distribution.
*  **ZERO-One Normalization**: Zero-one normalization subtracts the minimum pixel intensity and divides by the intensity range, scaling pixel values to the range 0 to 1.
*  **Percentile Normalization**: Percentile normalization subtracts a percentile value and divides by the difference between percentiles, scaling pixel values to the range 0 to 1.

## Feature Extraction 

*   **Edge Detection**
    *   **Roberts**: Detects edges by highlighting areas with rapid changes in pixel intensity.
    *   **Sobel**: Highlights areas with high first-order derivatives in both horizontal and vertical directions.
    *   **Scharr**: Provides better edge detection results, especially for images with high-frequency content.
    *   **Prewitt**: Similar to Sobel but with a slightly different kernel.
    *   **Canny**: Multi-stage algorithm for high-quality edge maps with minimal noise. It results in binary edge map.

*   **Corner Detection**
    *   **Harris Corner Detection**: Identifies corners by detecting large variations in intensity using a sliding window approach.
    *   **Shi-Tomasi Corner Detection**: Similar to Harris, with a different computation for the score value R, capable of finding the best corners in an image.

*   **Keypoint Detection**
    *   **SIFT (Scale-Invariant Feature Transform)**: Widely used for detecting corners, blobs, circles, and scaling images.
    *   **ORB (Oriented FAST and Rotated BRIEF)**: Utilizes FAST and BRIEF algorithms for one-shot facial recognition and keypoint matching based on intensity variations.

##  Segmentation
Image segmentation helps us identifying regions of interests in the scans.The methods used are:
*   **Multi-Otsu Thresholding**: Algorithm for segmenting pixels into multiple classes based on intensity. It determines several thresholds(based on input) to classify pixels into different intensity levels and returns threshold values, typically represented by a red line in the histogram.

*   **Region-based Segmentation**: Sobel filter generates elevation map highlighting regions. Markers are placed strategically to guide segmentation. Watershed algorithm segments image based on marker guidance. Fills holes in regions and labels connected components for comprehensive segmentation.

## Model Development 
For the model a custom Convolutional Neural Network(CNN) was implemented with weighted-cross entropy loss to handle data imbalance problem. The hyper-parameter tuned model achieved 85% accuracy on the test dataset. 


## Original Project Links
- Original Repository: [Link](https://dagshub.com/Omdena/TorontoCanadaChapter_BrainScanImages) 
- Dataset: [Link](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
- Streamlit app Deployment: [Link](https://huggingface.co/spaces/arpy8/Omdena_Toronto_Streamlit_App)


## Getting Started

* Open the Command line or Terminal .Clone this repository:

```
git clone https://github.com/Rick-Debjyoti/Alzheimer-s-image-classification.git
```

* Install dependencies:
```
pip install -r requirements.txt
```

* Move to the folder 
```
cd <folder name>
```

* Refer to the `src/` directory for code implementation of the Jupyter notebooks  for pre-processing and model-development.

* To open with VSCode ​
```
code .
```

* To open with jupyter notebook
```
jupyter-notebook
```

## License

This project is licensed under the [MIT License](LICENSE).
