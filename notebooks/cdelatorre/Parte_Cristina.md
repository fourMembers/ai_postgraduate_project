# <p align="center"> Pancreas tumor image segmentation </p>

### <p align="center"> UPC Artificial intelligence with Deep Learning Postgraduate Course </p>

##### <p align="center"> Students: Roger Borràs, Jaume Brossa, Cristina De La Torre, Cristian Pachón </p>

##### <p align="center"> Advisor: Santi Puch </p>

### Motivation

### About Medical Segmentation Decathlon

### Input dataset
The input dataset for this task consists in **282 3D volume images** (for training and validation) of size **512 x 512 x number of slices**, each 3D image is composed by a specific number of slices (around 100) of size 512x512. The **modality** used to take this images was **computed tomography (CT)**. This modality takes images of the body from several angles in order to compose the final 3D image of this part scanned, in our case a volume of the thorax, place where pancreas could be found. 

This images have been acquired from the **Memorial Sloan Kettering Cancer Center** (MSK), a highly recognized institute focused on cancer research.

<p align="center">
<img  src="https://github.com/fourMembers/ai_postgraduate_project/blob/master/notebooks/cdelatorre/ct__scan.gif?raw=true" alt="CT scanner" title="CT scanner"  width="400"> 
</p>

Normally in datasets like ImageNet, images have the *jpg* format or similar. However in this medical task,  the ***NifTI*** format is used. The NifTI format is commonly used in medical stuff, this format not only provide the image information (numpy array of the image) but also a header with metadata, where some relevant points were given. These relevant points consist mainly on the properties of the equipment where this images have been taken, data about the patient... This information has been fundamental during years to radiologist in order to segment and see the images properly, so it will be also relevant for us to deal with this task successfully.

### Purpose of our project
Our purpose during this project consists in **segment automatically ** from a raw CT image of the thorax, the **pancreas** and its **tumor**. One example of **what we had** and **what we wanted to achieve** is presented on the following image. As you can see below, we had a raw image difficult to visualize at first sight and its label.

<p align="center">
<img  src="https://i.imgur.com/fHGWnbJ.png" alt="Raw image and its respective label" title="Raw image and its respective label"  width="500" > 
</p>

At first, this images has been preprocessed applying some special techniques normally used in CT medical images for machine and deep learning problems. Then we customized an input pipeline for this types of 3D medical images to finally pass them to our customized network, also prepared for 3D input images.

The result that we wanted to obtain, consisted of a 3D image segmented. This output, as you can see in the above image, have three different colors. Each color correspond to one structure presented on the raw image:

 - Black, background
 - White, pancreas
 - Grey, pancreas tumor

So what we wanted during this project was to classify a 3D input image into this three different classes (background, pancreas and its tumor) and finally obtain a 3D output image with the pixels of this three different classes detected and labeled with its correspondent color.

### Challenges faced
As mention before, the input dataset consists of 3D images. It means that **a high amount of time** is needed to
train the model. Since the architecture of the net is quite complex, it took a lot of time **to make the model learn**. In this section, we describe what we did in order to solve these problems.

### Architecture
In this section, it is described the architecture of our proposal. This architecture is composed by two mainly parts:
 - **Input pipeline**,  process where images should be modified and prepared to enter on our network properly 
 - **Network**, process where our model will learn and predict the pancreas and its tumor segmentation

#### Input pipeline
As mentioned earlier, our input images consists of 3D CT medical images so we had to take this into account in order to manage them suitably. Before to start with the input pipeline, every CT medical image should be subjected to some changes, this changes are called **preprocessing**. Once images are preprocessed,  these are prepared in the **input pipeline** to enter in the network.

During preprocessing, various techniques has been carried out in order to obtain an **improved image** and ready to enter on the model. The **mainly changes applied to this images** are the following:

 - **Pixel spacing resampling**:
	 An special point taken into account was the pixel spacing in this 3D images. As you can see in the image below, we have the sizes of the pixels and the size of the voxel (distance between slices).  This pixel spacing depends on the scanner used, so in this way we could be dealing with images with different properties. It is important to keep the pixel spacing consistent, or else it may be hard for our network to **generalize**, so what we wanted was to have same sizes for all the images.

	We resampled all the images to a pixel spacing of type [1, 1, 1] . With this technique, a neural network will improve the predictions in different images  since they will have the same spatial information and the kernels can learn this information in the same way.

	<p align="center">
	<img  src="https://i.imgur.com/RK9sk3a.png" alt="(A) CT image (B) Image array (C) Voxel with sizes axbxc" title="(A) CT image (B) Image array (C) Voxel with sizes axbxcl"  width="400" > 
	</p>

 - **Windowing**:
		The intensities of grays in these types of medical images is not as usual as in normal images. Normal png files are usually in the standard range of 0-255 intensities values, while CT images can present  a high  range of intensities, from -1000 to 4000.
		
	This anormal range of intensities is due to the special unit of measurement in CT scans. This unit is the  **Hounsfield Unit**  **(HU)**  which is a measure of radiodensity. Each  **voxel**  (3D pixel) of a CT scan has an  **attenuation value**  that is the measure of the reduction intensity of a ray of light by the  **tissue**  through it passes. Each pixel is assigned to a numerical value called  **CT Number**, which is the average of all the attenuation values contained within the corresponding voxel. In the following image you will see this hounsfield scale, and some components or parts of the body are indicated, such as the pancreas.

	<p align="center">
	<img  src="https://i.imgur.com/XjwNd0j.jpg" alt="Hounsfield Units scale" title="Hounsfield Units scale"  width="800" > 
	</p>
	
	Since neither we nor the machines can recognize 4000 shades of gray easily, a technique called **windowing** has been used to limit the number of Hounsfield units that are displayed. For example if we want to examine the soft tissue in one CT scan we can use a **minimum pixel value** of -135 and a **maximum pixel value ** of 250, the tissues with CT numbers outside this range will appear either black or white. A narrow range provides a higher contrast.

	In our case, pancreas was our main interest. After some research we found that this organ was difficult to segment alone by this HU because there were more organs with similar intensity. But we could eliminate the intensities that were not interesting for us like the bones or the air, so finally we decided to use the soft tissue window explained above.

	<p align="center">
	<img  src="https://i.imgur.com/PgE70L3.png" alt="(Right) Histogram before applying windowing (Left) Histogram after applying windowing" title="(Right) Histogram before applying windowing (Left) Histogram after applying windowing" width="700" >
	</p>

	So thanks to **HU**, radiologist have been able during years to differentiate all the tissues presented in a CT scanner, and also we have been able to improve and **facilitate the learning of our model**.

 - **Normalization**: 
	Pixel intensity values has been normalized between values 0 and 1.

 - **Histogram equalization**:
	Histogram equalization is a very popular technique used in order to improve the appearance and contrast of medical images. Histogram equalization is a technique where the histogram of the resultant image is as flat as possible (image below). This allows for areas of lower local contrast to gain a higher contrast. 
	
	<p align="center">	
	<img  src="https://i.imgur.com/zcEnXmX.png" alt="(Rigth) Part of the histogram before applying equalization (Left) Part of the histogram after applying equalization" title="(Right) Part of the histogram before applying equalization (Left) Part of the histogram after applying equalization" width="700" > 
	</p>
		
Once all these steps were applied, we got an improved image (as you can see in the below image), and we were able to start with the **input pipeline**. We have created two different input pipelines depending on we were dealing with images of training or validation.

<p align="center">
<img  src="https://i.imgur.com/bbvS4DG.png" alt="(A) Raw image (B) Improved image" title="(A) Raw image (B) Improved image" width="500" > 
</p>

 ##### Training input pipeline 
<p align="center">
<img  src="https://i.imgur.com/tl74N1H.png" alt="Training input pipeline graph" title="Training input pipeline graph" >
</p>

This graph explain the flow of our training input pipeline. At first, we had the input image and its label, that were treated i the same way as numpy arrays. We divided these numpy arrays in different 3D volume patches of sizes 128x128x64, in this way we obtained small images and it was **easier to our model to learn**. 

This patches could include or not the pancreas and its tumor, so as we had the label we divided them into background patches and pancreas patches. This type of problem, as we have mentioned before is highly imbalance. So once we had the division, we applied **data augmentation** to the pancreas patches in order to increase the number of them until obtain 1.5 background patches for each pancreas patch. In this way we will improve the problem related with the imbalanced data.

<p align="center">
<img  src="https://i.imgur.com/sNDgHnh.png" alt="(Right) Image and label patch before transformations (Left) Image and label patch after transformations" title="(Right) Image and label patch before transformations (Left) Image and label patch after transformations" width = "300" >
</p>

This data augmentation has been made mild, in order to not disrupt the reality. Different techniques has been included as flips, elastic deformations, add noise and offset. Specially, elastic deformation was recommended in a lot of papers to increase the accuracy in this type of medical segmentation problems.

Once we had the background patches selected and the pancreas patches augmented, we concatenated them and did shuffle... and finally they were ready to train.

 ##### Validation input pipeline
<p align="center">
<img  src="https://i.imgur.com/T1AF0Qz.png" alt="Validation input pipeline graph" title="Validation input pipeline graph" width="700">
</p>

This graph explain the flow of our validation input pipeline. At first, we had the input image (that was also treated as  a numpy array). Such in the training input pipeline we divided this numpy array in different 3D volume patches, but in this case the size was bigger, 256x256x64. 

Once we divided the numpy array in patches, our input pipeline is ready to start the prediction.

#### Network
Our model was based in the **U-Net network**. The U-Net is a convolutional network architecture for fast and precise segmentation of images, and highly recommended for cases of medical imaging. We did this U-Net from scratch using tensorflow/keras and it was modified in order to be able to work with 3D images. 

<p align="center">
<img  src="https://i.imgur.com/oDbdCuY.png" alt="U-Net for pancreas image segmentation" title="U-Net for pancreas image segmentation" width = "500" >
</p>

This network architecture consists of a contracting path (left side) and an expansive path (right side), which gives it the u-shaped architecture. The contracting side,  is a typical convolutional network, that consists of repeated application of convolutions, followed by Rectified linear unit (ReLU) and max poolings. While during the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path.

### Iterations

### Final results
The hyperparameters from our last training where the following:


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



