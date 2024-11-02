# EE5179: Deep Learning for Imaging Project
 Image Restoration from Noisy & Blur Images while Preserving Defects of Interest




***
>In this report, we tackle the task of image denoising and deblurring using a
UNet-inspired model that incorporates elements of SUNet, a UNet variant enhanced
with Swin Transformer blocks. Our objective is to assess this architecture’s
performance on a validation set, with a focus on Peak Signal-to-Noise
Ratio (PSNR) and Structural Similarity Index (SSIM) metrics. This architecture
was selected for its capability to capture both local and global features
through a transformer-based approach, making it well-suited for high-quality
image restoration tasks.

## Test 
Ensure that the following packages are present on your system/environment:
1. torch 2.5.1+cu121
2. torchvision 0.20.1+cu121
3. numpy 1.26.4
4. cv2 4.10.0
5. mathplotlib
6. os
7. PIL
8. skimage

Once done, change the directories of the variables `img_folder`,`label_folder`, and `model_path` in `test.py` to the corresponding folders of the degraded images, clean images, and model weights respectively. To visualise your results, add a function towards the end of `test.py` to visualise results using `matplotlib`. Ensure that your images are 700 x 700 pixels in dimension.

Once the directories have been correctly specified, you can test the model by running,
```
python test.py
```
## Train  
To train the model on a different dataset, please go through the `Denoising_Model_code_final.ipynb` file. Ensure that your images are of a fixed dimension. 

## Result  

<img src = "https://i.imgur.com/golsiWN.png" width="800">  

## Visual Comparison  

<img src = "https://i.imgur.com/UeeOO0M.png" width="800">  

<img src = "https://i.imgur.com/YavgU0r.png" width="800">  





![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FSUNet&label=visitors&countColor=%232ccce4&style=plastic)  
