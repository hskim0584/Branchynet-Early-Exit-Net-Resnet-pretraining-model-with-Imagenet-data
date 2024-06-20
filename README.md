# Branchynet-Early-Exit-Net-Resnet-pretraining-model-with-Imagenet-data

## This Code is

to make pretrained model which works same as <torchvision resnet pretrained model>. 
But it is Branchy Net architecture


When applying branchnet to algorithms of other structures, most other algorithms often use the pretrained model provided by torchvision.


In such a situation, if you import and use the pretrained model provided by trochvision, learning will often fail.


The reason is that when using the branchnet structure, the model that adjusts the feature map size, called Inter Feature Extractor, does not have pretrained weights.

Resnet's pretrained model provided by torchvision has weights learned with ImageNet data. This code seeks to create a pretrained model that plays the same role as the pretrained model provided by torchvision in the resnet branchnet structure.

## Data
you can download Image Net Data for here
https://image-net.org/download.php

(you must sign up. Then, you can see download path.)
For play similar with torchvision pretrained models, download 
#### Image Net Large Scale Visual Recognition Challenge 2012(ILSVRC2012) data.

Because in torchvision, pretrained models are trained with ILSVRC2012 data.

This will take approximately 2-3 days... Good Luck !


## MODELs
For the model, only resnet was prepared.

If you want to apply the model to another algorithm after training, load and use the trained pth file here.
You can simply apply the entire model, but it may be helpful to load each layer separately.
 Therefore, when saving a model, we have prepared to save both each layer and the entire model.


## Requirements
pyton >= 3.8

```
pip install torch==2.2.1
pip tqdm==4.66.2
pip install torchvision==0.17.1
pip install torchsummary==1.5.1
```
## RUN
python main.py
