# CIFAR-10 classification
Welcome to my CIFAR-10 classification project! In this project, I use CNNs to classify images from the CIFAR-10 dataset. I consider CNNs of varying depth, starting with one layer, and increasing up to 6 layers. 
I analyze the added benefit of various techniques such as batch normalization, data augmentation, skip-connections, scheduled learning rates, etc. The final model achieves ~88% test set accuracy and is analyzed using Grad-CAM to understand what features it relies on, especially in challenging cases such as dog vs cat classification. I do an ablation study and discuss the trade-off of performance and model complexity.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/barbara-barta/CIFAR10-classification.git
cd CIFAR10-classification
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

### Notes

* Developed with Python 3.10
* Training performed on GPU 
* All hyperparameters can be modified in `configs.py`

## Project structure 
- models/ contains the folders of various models that were evaluated. Every folder contains a .tar parameter file, as well as hooked outputs, gradients and accuracies from each training epoch
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

The architecture is as follows:
<p align="center">
  <img width="800" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/network_diagram.png?raw=true" />
</p>
For training we use the Adam optimizer with a step learning rate scheduler with gamma = 0.5, and step size 10. We train on 150 epochs with a patience of 20 and use the cross entropy loss. In training, we use data augmentation consisting of a random crop, a horizontal flip and color jitter. 


## Results (TL;DR)
- Best test accuracy: **87.67%**
- Key techniques: data augmentation, batch normalization, skip connections, LR scheduling
- Findings: performance gains plateau with depth; data augmentation provides the largest boost
- **Key Insight:** In cat vs dog classification, the model relies heavily on localized facial features (snout), 
leading to poor robustness under occlusion or viewpoint changes.

## Evaluation 

We load the data and perform simple EDA to confirm that the classes are well balanced and that there is plenty of intra-class variation. We create the augmented and non-augmented data loaders and proceed to make simple one layer, two layer and four layer CNNs. We explore the effect of pooling and data augmentation. More details can be found in the notebook.
Next we create a simple 6 layer network with 6 convolutional layers of size 32, 32, 64, 64, 128, 128, ReLu activations and max pooling after the second and fourth convolutional layer. We train the network on 60 epochs with a patience of 10. The training loss curve shows signs of overfitting as the training loss increases, while the validation accuracy stagnates:
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/6layer_vanilla.png?raw=true" />
</p>
The final test accuracy of the model is 76.20%. 

Overfitting is amended by adding data augmentation to the model (model 1):
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/4layersDA.png?raw=true" />
</p>
As the model is exposed to a greater variety of data, and it no longer repeats training samples on each epoch, its ability to generalize increases. Using the same training parameters as in the previous model, the final test accuracy of the six layer network with data augmentation is 84.81%.
We plot the root mean square (RMS) of the outputs of each of the convolutional layers retrieved during training, plotted against the training batch. We make these plots to study the size of the outputs of each layer, that are then fed into the subsequent layer. We take the RMS as a measure of size, since it accounts for the number of features in each layer and the size of the input to each layer.
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model0_RMS.png?raw=true" />
</p>
We see that the output size seems to increase within the layers. Each subsequent layer is receiving inputs that are larger in the RMS sense. This is known as internal covariate shift: the input to each module has a different distribution. This leads to sub-optimal training. 

To correct this behavior, we add batch normalization to the model. We observe the following training loss behavior. 
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model1.png?raw=true" />
</p>
We see an expected added benefit of batch normalization: we reach 80% accuracy in about half the time as without batch norm. Also, the RMS plots also show that the layer inputs stay on the same scale for all layers. 
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/model1_RMS.png?raw=true" />
</p>
We make some further observations, such as the variability of the RMS decreasing through adding batch norm, and argue why that might be the case. More details can be found in the notebook. 

The next addition we explore is skip connections. Skip connections help when training deeper neural networks by allowing lower level information to flow more easily into the deeper parts of the network. With the addition of skip connections, the model achieves a test accuracy of 85.72%.
Finally, we add learning rate scheduling. Schedulers can help the model converge better by allowing it to take larger steps in the beginning and smaller steps as it gets closer to a minimum. The test accuracy is 85.96%.

We explore some other models, and then choose the best model. We do this by selecting the top 4 performing models, training each one 3 times for 100 epochs with a patience of 20. The six layer model with data augmentation, batch normalization, skip connections and a learning rate scheduler achieves the highest mean accuracy, 87,38%. Ideally we would run paired t-tests to prove that this model is actually better than the other ones, but given the limited sample size, the test will not have sufficient statistical power to yield meaningful conclusions. Given the small number of runs, we do not claim statistical significance. 
Instead, we select the model based on highest mean accuracy and consistent performance across runs.

## Best model analysis
We rerun the model for 150 epochs and with 20 patience. The final model achieves accuracy of 87,67%, a macro averaged F-score of 0.8773 and shows the following train accuracy curve:
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/final_model.png?raw=true" />
</p>
The curve shows convergence and only a minor sign of overfitting, as the train and validation accuracy remain within 5% of each other. Following is a plot of the confusion matrix with the percentage of misclassified samples per class:
<p align="center">
  <img width="400" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/cm.png?raw=true" />
</p>
Classes that are most often correctly classified are automobile, ship and truck. Classes that the model struggles the most with are dog, cat and bird. This makes sense intuitively, since vehicles have simpler textures, and their edges are non-deformable, while animals can change shape more drastically depending on the position they take. The biggest confusion seems to be between the classes 'cat' and 'dog'. We dive deeper into this issue by making grad-CAM plots. Gradient-weighted Class Activation Mapping (Grad-CAM) is an explainable AI technique that creates heatmaps to visualize what regions of an image a CNN-based model focuses on to make predictions. By using gradients of a target concept flowing into the final convolutional layer, it highlights the most important features in the image. 
We first plot grad-CAMs of the images of dogs which were correctly classified. 
<p align="center">
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/snout_dog.png?raw=true" />
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/snout_dog2.png?raw=true" />
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/snout_dog3.png?raw=true" />
</p>
It seems that the model mainly focuses on the dog's face, particularly the snout region. We then plot grad-CAMs of images of dogs that were misclassified as cats.
<p align="center">
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/no_snout_dog.png?raw=true" />
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/no_snout_dog2.png?raw=true" />
  <img width="250" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/red_dog.png?raw=true" />
</p>
In many of these images, the snout is not recognizable, either due to a strange angle, or due to low contrast between the snout and the color of the hair around it.
It seems that the model has learned a shortcut feature for “dog” — it heavily relies on snout/face-specific patterns. When that feature is clear, it succeeds; when it is obscured (angle, lighting, contrast), performance drops. The CNN has insufficient robustness to viewpoint + appearance variation. It likely under-utilizes global shape, texture and color. Take for example the last two misclassified samples. That reddish fur color is much more common in dogs than in cats, yet the model does not recognize this.

## Limitations and Future work
As described in the previous section, the model heavily focuses on one feature - the snout. One possible solution to this problem would be to add data augmentation that specifically targets this feature over-reliance. One example is the 'Cutout' data augmentation which randomly masks square regions of the input image during training. Another regularization strategy for reducing feature over-reliance is the 'CutMix' augmentation. Here, patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches. This way, we might end up with a training image that contains the head of a cat, but the body of a dog. Trained on this example, the classifier use the cat's head to make a prediction of the label 'cat', and the dog's feet to make a prediction of a 'dog' label. The technique is similar to the 'Cutout' technique, only it uses the training pixels more efficiently since we are not replacing a random part of the image with black pixels. 

Another improvement could come through changing the architecture. Aside from adding more layers and increasing the channel width, we could adjust the way skip connections are used. As mentioned before, it seems that the network is not using texture cures enough. Adjusting the skip connections would allow the network to better combine the low- and high-level features. In the current set-up, we use max-pooling after adding the identity path to the convolution path. As these paths are pooled together, some of the low-level information likely gets lost. Conversely, in famous ResNet architectures the downsampling occurs before adding the identity and the convolution path. This allows the low level information to flow into the deeper parts of the network with less alteration. 

## Ablation study
Below is an ablation study of the model. The results were obtained by training each model 3 times on 100 epochs and taking the average test set accuracy.
| Model  | Accuracy |
| ------------- | ------------- |
| baseline 6-layer network | 73,63%  |
| + DA  | 85,77%  |
| + BatchNorm  | 86,22%  |
| + Skip connections  | 84,23%  |
| + lr scheduler  | 87,38%  |

There is a drop in performance after adding skip connections. It is possible that skip connections do not contribute to performance at this depth, and the improvements we see in the final model actually come from the addition of the learning rate scheduler, rather than the skip connections - something perhaps worth investigating.

## Impact of parameters on performance
<p align="center">
  <img width="700" alt="image" src="https://github.com/barbara-barta/CIFAR10-classification/blob/main/figures/parameters.png?raw=true" />
</p>
The following plot shows the accuracy of several models plotted against the number of parameters in the model. We consider a one layer network, a two layer network, two four layer networks (one with data augmentation and one without) and two six layer networks (the 'simple' six-layer network with only data augmentation, and the final six layer network we chose as the best model). When we compare the one and two layer network performance with the four layer network performance, the difference is stark: the simpler networks achieve only ~60% accuracy, while the simple four layer network without data augmentation achieves ~74% accuracy. Adding data augmentation brings this number up to ~83%, with no additional parameter cost. 
In comparison, increasing the number of layers to six gives modest gains, with the highest performing model attaining ~87% accuracy. This brings into question the added value of making a more complex model, when the training and evaluation costs increase faster than performance does. 

## Acknowledgements/References
### Libraries
- PyTorch — model training
- NumPy — numerical operations
- scikit-learn — evaluation metrics
- Matplotlib — visualization
### Datasets, articles
- CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html  
  Used for training; images normalized and augmented.
- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers
with Localizable Features*, 2019  
  [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1905.04899)
- *Implementing Grad-CAM in PyTorch* by Stepan Ulyanin — [Medium article](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
