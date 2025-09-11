# A Study of Image Inpainting Methods via Various Perspective Image Synthesis

This repository contains the official source code for the paper: "A Study of Image Inpainting Methods via Various Perspective Image Synthesis Using CNN-based and Diffusion-based Models," presented at CVGIP 2025.

**Authors:** En Chen (陳恩), Yi-Hsin Chiang(姜義新), Chih-Chang Yu (余執彰), Hsu-Yung Cheng(鄭旭詠)

[![CVGIP 2025](https://img.shields.io/badge/CVGIP-2025-blue.svg)](https://ippr.org.tw/cvgip/cvgip_previousinformation/)

---

### Table of Contents
1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Methodology](#methodology)
4.  [Dataset](#dataset)
5.  [Results](#results)


---

## Overview

Image inpainting aims to restore missing or corrupted regions in an image. Current deep learning models often generate plausible but not necessarily faithful reconstructions, as they typically rely on surrounding pixels or text prompts.

This project addresses this limitation by proposing a novel two-stage inpainting method. Our approach leverages a supportive image from a different viewpoint of the same scene to provide explicit guidance for the inpainting process. This image-guided conditional input serves as a strict constraint, leading to less ambiguous and more accurate restorations that align with the original scene.

We validate our method on both a CNN-based model (HiFill) and a diffusion-based model (Latent Diffusion Model), demonstrating that our preprocessing step enhances the performance of both architectures.

## Key Features

* **Two-Stage Inpainting Pipeline:** A preprocessing stage generates a coarse inpainting result, which then guides a fine-tuning inpainting model.
* **Multi-View Synthesis:** Utilizes a reference image from a different perspective to accurately fill in missing content.
* **Hybrid Model Support:** The method is tested and proven effective on two major inpainting architectures:
    * **CNN-based (HiFill):** The refine network is adapted to work with our preprocessed input.
    * **Diffusion-based (LDM):** The U-Net is fine-tuned using LoRA, with the preprocessed image serving as an additional conditional input.
* **Faithful Reconstruction:** By providing strong, spatially-aligned guidance, our method excels at restoring specific objects within the masked region, which other models might omit.

## Methodology

Our pipeline consists of two main stages.

### Stage 1: Preprocessing for Coarse Synthesis

1.  **Feature Matching:** We use the **LOFTR (Local Feature Matching with Transformer)** module to find a set of robust matching points between the target image (with a masked region) and the reference image.
2.  **Homography Estimation:** A homography matrix `H` is estimated from the matched points using RANSAC to ensure geometric consistency. To improve alignment, we prioritize matching points located within the masked region.
3.  **Pixel Overlay:** The estimated homography matrix is used to warp the reference image and overlay the corresponding pixels onto the masked region of the target image. This creates a coarsely inpainted result that serves as a strong prior for the next stage.

### Stage 2: Model-based Inpainting

The coarsely synthesized image from Stage 1 is fed into a modified inpainting model.

* **For the HiFill-based model:** We replace the original coarse network with our preprocessing step and feed the result directly into the refine network for enhancement.
* **For the LDM-based model:** We concatenate the latent representations of the coarsely inpainted image and the binary mask with the noise input. The diffusion U-Net is then efficiently fine-tuned using Low-Rank Adaptation (LoRA) layers to process this new input format.

## Dataset

We used the **Image Matching Challenge 2022** dataset from Google Research. This dataset is ideal as it contains images of various scenes, such as the British Museum, captured from multiple viewpoints, which is essential for our reference-based approach.


## Results
Our proposed method demonstrates significant improvements over baseline models, especially in faithfully reconstructing missing objects.

Quantitative Results:

Our method achieves superior performance across PSNR, SSIM, and LPIPS metrics.

Our HiFill-based model achieved a PSNR of 33.1101, a 2.93% improvement over the same model trained without our preprocessed input.

Our LDM-based model achieved a PSNR of 27.9391, showing a 1.93% improvement over its baseline counterpart.
