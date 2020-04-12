# <p align="center"> Pancreas tumor image segmentation </p>

### <p align="center"> UPC Artificial intelligence with Deep Learning Postgraduate Course </p>

##### <p align="center"> Students: Roger Borràs, Jaume Brossa, Cristina De La Torre, Cristian Pachón </p>

##### <p align="center"> Advisor: Santi Puch </p>

### Motivation

All of us are interested in biomedical applications and a central problem in health is the patient diagnostic process. This figure represents the general pipeline for patient diagnosis in health care:

<p align="center">
    <img align="center" img width="460" height="300" src="images/motivation_data/dp.png">
</p>

* A first step, the engagement where a patient experiences a health problem.
* Second, define the problem with medical data collection and interpretation.
* And finally, treatment and evaluate the outcomes.

For the data collection in patient diagnosis, there are different data sources like clinical examination, health monitoring, laboratory results and medical imaging. 

<p align="center">
    <img align="center" img width="460" height="300" src="images/motivation_data/md.png">
</p>

Medical imaging is a very important tool in diagnosis, and it will be more important in the future without any doubt. Our project is focused on apply machine learning in this kind of data to improve diagnosis. In a recent paper from New England journal of Medicine (NEJM), they suggest that neural networks has the potential to revolutionize health in the future.

<p align="center">
    <img align="center" src="images/motivation_data/NEJM_1.png">
</p>

### About Medical Segmentation Datathon

To practice this type of problems, we searched data from this medical segmentation dechatlon competition  and we picked the pancreas tumor segmentation problem. 

<p align="center">
    <img align="center" src="images/motivation_data/MSD.png">
</p>

<p align="center">
[Medical Segmentation Decathlon]( http://medicaldecathlon.com/)
</p>


### Input dataset

The data available is from memorial Sloan Kettering cancer center in NY. To train the model we have:
* 420 3D CT scans in total
* 282 training and 139 for testing.

As a prior challenge, this is a class unbalance problem because the tumor is a very small region compared the overall CT scan. 

### Challenges faced
As mention before, the input dataset consisted of 3D images. It meant that **a high amout of time** was needed to
train the model. Since the architecture of the net was quite complex, it took a lot of time **to make the model learn**. In this section, it is described the actions done in order to solve these problems.

#### High amout of time
It was observed the model needed a large amount of time to train. The first hyphotesis was the images were too big. In order to validate this hypotheis, we did the following changes:

* Resizing: It actually lowered the time needed but at the same time, a lot of information was lost. So, we decided not to use it

<p align="center">
    <img align="center" src="images/challenges_faced/resize.png">
</p>

* Crop: Image crop along axes x,y and z. Relevant information was kept and input was smaller. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/crop.png">
</p>

* Patches: Divide the image into patches. It helped tp solved computational problems. We decided to used it

<p align="center">
    <img align="center" src="images/challenges_faced/patches.gif">
</p>

#### Making the model learn
It was observed that the model was not learning, since it was assigning all the pixels to the backgroud class. After analysing the dataset, we saw the data was highly unbalanced. The following figures shows the raw image and the cancer:

<p align="center">
    <img align="center" src="images/challenges_faced/raw_image.png">
</p>

<p align="center">
    <img align="center" src="images/challenges_faced/target.png">
</p>

As we can see, pancreas tumor is a really small portion of the image. We tested three hypothesis in order to see if the model was able to learn. These were:

* Use a problem-oriented loss function. A lot of papers saig the Generalised Dice Loss (GDL) was the indicated one to use for this problem. After using GLD the model started learning. The formula of this loss is the following one:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=1 - 2 \frac{\sum_{}^{2}w_l\sum_{n}^{}r_{ln}p_{ln}}{\sum_{}^{2}w_l\sum_{n}^{}\big(r_{ln} + %2B p_{ln}\big)}">

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{1}{ \big(\sum_{n}^{}r_{ln} \big)^2 }">
</p>

* Balance the data in order the have the same amount of pancreas as backgraud. This also helped the model to learn

* Data augmentation: In order to generate different samples of pancreas. Even though it was not necessary (becuase there were a lot of images), we implemented it to learn how to do these transformations.

### Architecture

### Iterations
In order to make small but relevant improvements, several iterations have been done. In this section, they are described.

##### Overfitting
The first step done was make the model overfit. To achieve it, the following configuration was used:

* Used Binary Cross Entropy Loss
* Used only 15 patches for training, 5 for validation
* Trained for around 370 epochs (9 hours)

The following figure shows the traning and validation loss:

<p align="center">
    <img align="center" src="images/iterations/loss_overfit.png">
</p>

We can see that from epoch 200 the training loss is still descending but the validation loss is flat. 

The following figures show how the model was overfitted. The firt figure is a prediction for an image from the traning set and the second one is a prediction for an image from the validation set:

<p align="center">
    <img align="center" src="images/iterations/train_overfit.png">
</p>

<p align="center">
    <img align="center" src="images/iterations/validation_overfit.png">
</p>

Clearly, the model is memorising the training set and, thus, it is unable to predict anything for an unseen image, since everything is assigned to background class.

#### Using Generalised Dice Loss
After building an overfitted case, the loss function was changed. The following configuration was used:

* Used Generalized Dice Loss, which optimum value in our implementation is -1
* Used only 15 patches for training, 5 for validation
* Trained for around 230 epochs (5 hours)

The following figure shows the training and the validation loss function:

<p align="center">
    <img align="center" src="images/iterations/loss_dice.png">
</p>

The model is slowly learning. In this scenario there is no overfit, as we can see in the following figures:

<p align="center">
    <img align="center" src="images/iterations/train_dice.png">
</p>

<p align="center">
    <img align="center" src="images/iterations/validation_dice.png">
</p>

The first figures are an image from the traning set and another for the validation set (respectively).

#### First trial with the whole dataset
After having all the ingredients ready, we performed an iteration using (almost) all the images. The following configuration was used:

* Used Generalized Dice Loss
* Used 280 images for training, 70 for validation
* Used balanced data input + data augmentation (just for fun!!)

This configuration provided the following loss function:

<p align="center">
    <img align="center" src="images/iterations/loss_first.png">
</p>

Everything looked fine, but there was an **error**. It was used the same input pipeline (balanced classes + data augmentation) for training and for validation. It was **fixed** using a normal input pipeline for validation without data augmentation: more realistic strategy for inference.

After fixing it, we were ready to perform the final iteration. It is described in *final results* section.

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

<p align="center">
     <img align="center" src="images/final_results/final_losses.png">
</p>
