# Image-Colorization-Using-Conditional-GANs


Colab Notebook - https://colab.research.google.com/drive/1xog3aiiXB1wXAGJ14rqhATp0fedGWCdl?usp=sharing

ResNet based Unet - https://drive.google.com/file/d/1-Clo-DlyCzR0jHlC-MgKRqKXr4bwmHsQ/view?usp=sharing

Final Trained Model - https://drive.google.com/file/d/1eMAmWVrhoBudzJ0y6MXWOnA6iwLhotMd/view?usp=sharing


## 1- Problem Statement

The task of colorizing black and white photographs necessitates a lot of human input and hardcoding. The goal is to create an end-to-end deep learning pipeline that can automate the task of image colorization by taking a black and white image as input and producing a colorized image as output.


## 2- Motivation

Colorization is the process of adding color information to monochrome photographs or videos. The colorization of grayscale images is an ill-posed problem, with multiple correct solutions. Online tools can be used for image colorization but the problem with these tools is a lack of inductive bias which results in inappropriate colors and doesn’t even work for a few image domains. Deep learning algorithms that better understand image data like the colors that are generally observed for human faces should ideally perform better in this task.


## 3- Solution
- The colorization of grayscale images can be thought of as an image-to-image translation task where we have the corresponding labels for the input grayscale image. A conditional GAN conditioned on grayscale images can be used to generate the corresponding colorized images.
- The architecture of the model consists of a conditional generator with grayscale image inputs and a random noise vector and the output of the generator are two image channels a, b in the LAB image space to be concatenated with the L channel i.e. the grayscale input image.
- This generated image is input to the PatchGAN discriminator which outputs a score for each patch in an input image based on if the patch is real or not. These are used as learning signals for the Generator to generate better images. Along with the generated images, the Discriminator is also fed real images.
- When trained adversarially, the generator should get better at generating realistic colorized images that share a similar structure with the input grayscale images and the discriminator should get better at discriminating between real and fake images.
- The trained generator can then be used for generating colorized images given input grayscale images.


## 4- Dataset Used

The original paper uses the whole ImageNet dataset but here I have used a subset of randomly selected RGB images (roughly 10000 images) to be used for training from the COCO images dataset.
So the training set size is 0.6% of what was used in the paper! 

The Dataset is split into 2 parts - 8000 training images and 2000 validation dataset.


## 5- Library Used

Pytorch

Tensorboard

Fastai


## 6- Setup and Folders

The Colab notebook is publicly available. To run the Tensorboard - click on the refresh button on the inline tensorboard to show the progress of training in real time and also the fake and real images produced at each time step/100. 

The repository consists of main code in google collab notebook, the outputs produced by the final model and the input black and white images, also I have provided the pre-trained Resnet based U-net and the final model as saved pytorch(pickle) file from drive folder. I did not upload to the github as the files sizes are quite large.


## 7- Input Images/Data Format 

The input image will be in LAB color space because if we input a RGB image, we have to first convert the image to grayscale, feed the grayscale image to the model and hope it will predict 3 numbers, which is a way more difficult and unstable task due to the many more possible combinations of 3 numbers.
In L*a*b color space, there’s three numbers for each pixel but these numbers have different meanings. The first number (channel), L represents a black and white image. The a and b channels encode how much green-red and yellow-blue each pixel is, respectively, hence, the color part.


## 8- Training Strategy

The model is trained adversarially - in a cGAN architecture - UNet as generator and PatchGan Discriminator. 

In this approach two losses are used: L1 loss, which makes it a regression task of predicting 2 channels, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner by assigning the outputs a number(probability) indicating how "real" they look..

The Code is divided into 2 sections - first I have implemented the task as it is given in the pix2pix paper, which will be our baseline model and in 2nd part - I have used pre-trained ResNet model as backbone to build the UNet - resulting in a much better results in less epoch.


## 9- Results
The baseline model works as expected and as can be seen in the loss function graphs - the generator keeps getting better and better in producing real like images and the discriminator gets better at recognizing fake images from real ones. the loss becomes somewhat stable at around 6-7 epochs but interestingly the models keep getting better in producing real like images until much later.

The baseline model would require 30 or more epochs to train and produce good quality images. As can be seen, there are some patches of color in the output, or some dark/gray area. This is maybe due to the fact that neither the generator nor the discriminator knows anything about the images at the starting phase- Blind training the blind.

Hence, in case of cGANs it is better to evaluate the performance in terms of output produced than just the loss trajectory.
The cGAN model with ResNet was a significant improvement from the baseline model - It was pre-trained on the training set for about 10 epochs and then in just 15 epochs produced  much better quality images than the baseline model, saving both time and effort.  The patches of color are much less at just 15 epochs and the model should produce near real like images at just 20 epochs!!

The model output and original black and white images input for the 2nd model are provided in the repository for reference. 


## 10- References 

This project was done as part of Summer Projects under VLG , IIT Roorkee. I have referred to below awesome resources and materials available online.

1- For learning pytorch and related references =  https://pytorch.org/tutorials/ -  

2- Overview of GANs - https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09

3- Conditional GANs - https://jonathan-hui.medium.com/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d

4- LAB Color Space - http://shutha.org/node/851

5-Pix2Pix Image Translation Paper - https://arxiv.org/pdf/1611.07004.pdf

6- The official github repository of pix2pix paper - https://github.com/phillipi/pix2pix

7- For Unet Architecture details - https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

8- A nice blog on pix2pix GAN models - https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

9- Fast Ai tutorial for making pre-trained models -  https://docs.fast.ai/tutorial

10- Tensorboard Tutorials - https://www.tensorflow.org/tensorboard


## 11- Future Improvements

1-  Due to limited setup , I have trained for about 6 epochs on the 1st model and about 15 epochs with 20 pre-training on the 2nd model. In future, to see much better results I would suggest running the 1st model on more epochs,at least around 30-40 and about 20 epochs on the 2nd model.

2- Experiment with dropout layers , changing the architecture of UNet or discriminator or using different datasets with more data.
