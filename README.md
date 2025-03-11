<div align="center">
  
# MINE: Uncertain Multimodal Intention and Emotion Understanding in the Wild

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/CVPR-2025-6790AC.svg)](https://cvpr.thecvf.com/)
[![Paper](http://img.shields.io/badge/Paper-6720AC.svg)](https://cvpr.thecvf.com/)

</div>

## Updates

<!-- :satisfied: (03/11/2024) Code Released! -->
- :blush: (02/27/2025) Paper Accepted!

## Abstract
Understanding intention and emotion from social media poses unique challenges due to the inherent uncertainty in multimodal data, where posts often contain incomplete or missing modalities. While this uncertainty reflects real-world scenarios, it remains underexplored within the computer vision community, particularly in conjunction with the intrinsic relationship between emotion and intention. To address these challenges, we introduce the **M**ultimodal **I**ntentio**N** and **E**motion Understanding in the Wild (**MINE**) dataset, comprising over 20,000 topic-specific social media posts with natural modality variations across text, image, video, and audio. MINE is distinctively constructed to capture both the uncertain nature of multimodal data and the implicit correlations between intentions and emotions, providing extensive annotations for both aspects. To tackle these scenarios, we propose the Bridging Emotion-Intention via Implicit Label Reasoning (BEAR) framework. BEAR consists of two key components: a BEIFormer that leverages emotion-intention correlations, and a Modality Asynchronous Prompt that handles modality uncertainty. Experiments show that BEAR outperforms existing methods in processing uncertain multimodal data while effectively mining emotion-intention relationships for social media content understanding. Dataset and code will be released.
