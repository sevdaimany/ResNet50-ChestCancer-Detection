
# ResNet50-ChestCancer-Detection

This project implements the ResNet50 architecture from scratch and utilizes it for classifying chest cancer 
using the Chest CT-scan Images dataset available on Kaggle.

## Project Highlights

- implementation of ResNet50 from scratch.
- Transfer learning using the ResNet50 model as a backbone.
- Classification of chest CT scan images.
- Monitoring and visualizing training and validation accuracy.


## Introduction to ResNet
![resnet50](https://github.com/sevdaimany/ResNet50-ChestCancer-Detection/blob/master/resnet.png)


Residual Networks, or ResNets, are a type of deep neural network architecture that was introduced to address the 
vanishing gradient problem in very deep networks. ResNets achieve this by using skip connections, also known as 
shortcut connections, to allow the gradients to flow directly through the network, making it easier to train 
very deep networks.

The core idea behind ResNet is the residual block. Instead of learning the desired output, these blocks learn a 
residual or the difference between the desired output and the current output. By stacking multiple residual 
blocks, deep networks can be trained more effectively.

## Dataset

The dataset used for this project is the Chest CT-scan Images dataset, which can be found 
[here](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images).

## Model Evaluation

I have trained and evaluated the ResNet50-based model on the Chest CT-scan Images dataset. Below is a plot 
illustrating the accuracy of the model on both the training and validation datasets:

![Training and Validation Accuracy 
Plot](https://github.com/sevdaimany/ResNet50-ChestCancer-Detection/blob/master/train_val_plot.png)

The x-axis represents the training epochs, while the y-axis represents the accuracy.


## Additional Resources

- If you want to learn more about the ResNet architecture, you can read this [article on 
ResNet](https://arxiv.org/abs/1512.03385).


