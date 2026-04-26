# CIFAR-10 classification
Welcome to my CIFAR-10 classification project! In this project, I use CNNs to classify images from the CIFAR-10 dataset. I consider CNNs of varying depth, starting with one layer, and increasing up to 6 layers. 
I analyze the added benefit of various techniques such as batch normalization, data augmentation, skip-connections, scheduled learning rates etc. The final model achieves ~88% test set accuracy and is analyzed using Grad-CAM to understand what features it relies on, especially in challenging cases like dog vs cat classification. I also  discuss the trade-off of performance and model complexity.

## Setup
To begin, install the packages from requirements.txt 

## Project structure 
- models/ contains the folders of various models that were evaluated. Every folder contains a .tar parameter file,
- model.py contains the CNN architecture 
- dataset.py contains the CIFAR-10 loading + augmentation
- train.py contains the training loop + checkpointing
- evaluate.py contains the evaluation + confusion matrix
- utils.py contains helper functions (metrics, plotting)
- configs.py contains hyperparameters and settings
- img_classification.ipynb is the notebook containing the experiments

## Final model  
The final model architecture contains:
- 6 convolutional layers
- 3 residual-style blocks (intra-block skip connections)
- Batch Normalization after each convolution
- ReLU activations
- MaxPooling between blocks
- Fully connected classifier

The structure is as follows:
1. input: 3 channels
2. 32 channels: [Conv → BN → ReLU] ×2 + Skip → Pool
3. 64 channels: [Conv → BN → ReLU] ×2 + Skip → Pool
4. 128 channels: [Conv → BN → ReLU] ×2 + Skip
5. → FC → 10 classes

For training we use the Adam optimizer with a step learning rate scheduler with gamma = 0.5, and step size 10. We trian on 150 epochs with a patience of 20 and use the cross entropy loss. In training, we use data augmentation consistic of a random crop, a horizontal flip and color jitter. 

## Evaluation 

We load the data and perform simple EDA to confirm that the classes are well balanced and that there is plenty of intra-class variation. We create the augmented and non-augmented data loaders and proceed to make simple 1 layer and 2 layer CNNs. We explore the effect of pooling. We then create a four layer CNN, on which we explore the effect of data augmentation. More details can be found in the notebook.
Next we create a simple 6 layer network with 6 convolutional layers of size 32, 32, 64, 64, 128, 128, ReLu activations and max pooling after the second and fourth convolutional layer. The training loss curve shows signs of overfitting as the trianing loss increases, while the validation accuracy stagnates:of over
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/6layer_vanilla.png?raw=true" />
</p>
The final test accuracy of the model is 76.20%. 

Overfitting is ammended by adding data augmentation to the model:
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/4layersDA.png?raw=true" />
</p>
As the model is exposed to a greater variety of data, and it no longer repeats training samples on each epoch, its ability to generalize increases. The final test accuracy of the four layer network with data augumentation is 84.81%.
We plot the root mean square (RMS) of the outputs of each of the convolutional layers retrieved during training, plotted against the training batch. We make these plots to study the size of the outputs of each layer, which are then fed into the subsequent layer. We take the RMS as a measure of size, since it accounts for the number of features in each layer and the size of the input to each layer.
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model0_RMS.png?raw=true" />
</p>
We see that the output size seems to increase within the layers. Each subsequent layer is receiving inputs which are larger in the RMS sense. This is known as internal covariate shift: the input to each module has a different distribution. This leads to sub-optimal training. 

To correct this behaviour, we add batch normalization to the model. We observe the following training loss behaviour. 
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model1.png?raw=true" />
</p>
Note that we reach 80% accuracy in about half the time as without batch norm. The RMS plots also show that the layer inputs stay on the same scale for all layers. 
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model1_RMS.png?raw=true" />
</p>
The next addition we explore is skip connections. The model achieves a test accuracy of 85.72%. Wwe make some further observations, such as the variability of the RMS decreasing through adding batch norm, and argue why that might be the case. More details can be found in the notebook. 

Finally, we add learning rate scheduling. Schedulers can help the model converge better by allowing it to take larger steps in the beginning and smaller steps as it gets closer to a minimum. The test accuracy is 85.96%.

We explore some other models, such as one with reduced number of pooling layers. We then chose the best model. We do this by selecting the top 4 performing models, training each one 3 times for 100 epochs with a patience of 20. We compute the confidence interval of the accuracy of each model. One model has a strictly higher accuracy CI than the rest of the models. This is the model with batchnorm, skip connections and learning rate scheduling. 

## Best model analysis

## Limitations and Future work

## Ablation study

## Impact of parameters on performance

## Acknowledgements/References

