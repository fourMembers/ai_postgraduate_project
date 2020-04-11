# <p align="center"> Pancreas tumor image segmentation </p>

### <p align="center"> UPC Artificial intelligence with Deep Learning Postgraduate Course </p>

##### <p align="center"> Students: Roger Borràs, Jaume Brossa, Cristina De La Torre, Cristian Pachón </p>

##### <p align="center"> Advisor: Santi Puch </p>

### Motivation

### About Medical Segmentation Datathon

### Input dataset

### Challenges faced
As mention before, the input dataset consisted of 3D images. It meant that **a high amout of time** was needed to
train the model. Since the architecture of the net was quite complex, it took a lot of time **to make the model learn**. In this section, it is described the actions done in order to solve these problems.

#### High amout of time
It was observed the model needed a large amount of time to train. The first hyphotesis was the images were too big. In order to validate this hypotheis, we did the following changes:

* Resizing: It actually lowered the time needed but at the same time, a lot of information was lost. So, we decided not to use it

<p align="center">
    <img align="center" src="images/challenges_faced/resize.png">
</p>

* Crop: Image crop along axes x,y and z. Relevant information was kept. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/crop.png">
</p>

* Patches: Divide the image into patches. It helped tp solved computational problems. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/patches.gif">
</p>

#### Making the model learn
It was observed that the model was not learning. The model was assigning all the pixels to backgroud class. After analysing the dataset, we saw the data was high unbalanced. The following figures shows the raw image and the cancer:

### Architecture

### Iterations

### Final results
The hyperparameters from our last training were the following:


|          Hyperparameter         |    Value   |
|:-------------------------------:|:----------:|
|           Architecture          |  3D U-Net  |
|      Initial learning-rate      |    0.001   |
|            Batch size           |      2     |
|              Epochs             |     78     |
|         Train patch size        | 128x128x64 |
|      Validation patch size      | 256x256x64 |
|        # images training        |     120    |
|       # images validation       |     30     |
| Background/Pancreas patch ratio |     1.5    |

![alt text](https://github.com/fourMembers/ai_postgraduate_project/blob/master/images/final_results/final_losses.png)



