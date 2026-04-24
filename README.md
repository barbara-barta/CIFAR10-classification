# CIFAR-10 classification
Welcome to my CIFAR-10 classification project! In this project, I use CNNs to classify images from the CIFAR-10 dataset. I analyze the added benefit of various techniques such as batch normalization, data augmentation, skip-connections, 
scheduled learning rates etc. The final model achieves ~88% test set accuracy and is analyzed using Grad-CAM to understand what features it relies on, especially in challenging cases like dog vs cat classification. I also  discuss the trade-off of performance and model complexity.

## Model architecture 
The final model contains:
- 6 convolutional layers
- 3 residual-style blocks (intra-block skip connections)
- Batch Normalization after each convolution
- ReLU activations
- MaxPooling between blocks
- Fully connected classifier

Structure
1. input: 3 channels
2. 32 channels: [Conv → BN → ReLU] ×2 + Skip → Pool
3. 64 channels: [Conv → BN → ReLU] ×2 + Skip → Pool
4. 128 channels: [Conv → BN → ReLU] ×2 + Skip
5. → FC → 10 classes

