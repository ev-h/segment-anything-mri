# segment-anything-mri
Segment Anything Model (SAM) applied to MRI

![feature_image](notebooks/images/featuredImage.png)

The SAM_demo_cardiac_MRI.ipynb notebook demonstrates how to apply Segment Anything Model (SAM) in medical image analysis. Developed by Meta AI Research, SAM was trained on an enormous dataset of 11 million images of everyday objects and scenes to predict high-quality segmentation masks. While SAM was not trained on medical images, its zero-shot capability can be utilized for medical imaging applications, potentially mitigating data scarcity problems faced by many medical practitioners and researchers. In this notebook, we will learn how to load a pretrained SAM, read and preprocess a DICOM image, predict a segmentation mask, and evaluate the segmentation.