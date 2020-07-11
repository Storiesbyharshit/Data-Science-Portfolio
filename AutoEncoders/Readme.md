# Autoencoders

Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning.
Specifically, we'll design a neural network architecture such that we impose a bottleneck in the network which forces a compressed knowledge representation of the original input. 
If the input features were each independent of one another, this compression and subsequent reconstruction would be a very difficult task. 
However, if some sort of structure exists in the data (ie. correlations between input features), this structure can be learned and consequently leveraged when 
forcing the input through the network's bottleneck.

## Autoencoder Components:
#### Autoencoders consists of 4 main parts:

1- **Encoder**: In which the model learns how to reduce the input dimensions and compress the input data into an encoded representation.

2- **Bottleneck**: which is the layer that contains the compressed representation of the input data. This is the lowest possible dimensions of the input data.

3- **Decoder**: In which the model learns how to reconstruct the data from the encoded representation to be as close to the original input as possible.

4- **Reconstruction Loss**: This is the method that measures measure how well the decoder is performing and how close the output is to the original input.

_**The training then involves using back propagation in order to minimize the network’s reconstruction loss.**_

<p align="center">
  <img alt="For-the-Badge-Python" src="https://blog.keras.io/img/ae/autoencoder_schema.jpg">
  
</p>


Autoencoder is a neural network (NN), as well as an un-supervised learning (feature learning)
algorithm. Autoencoders are an unsupervised learning technique in which we leverage neural
networks for the task of representation learning. Specifically, we'll design a neural network
architecture such that we impose a bottleneck in the network which forces a compressed
knowledge representation of the original input. If the input features were each independent of
one another, this compression and subsequent reconstruction would be a very difficult task.
However, if some sort of structure exists in the data (ie. correlations between input features), this
structure can be learned and consequently leveraged when forcing the input through the
network's bottleneck.

• It applies backpropagation, by setting the target value same as input.

• It tries to predict x from x, without need for labels.

• It tries to learn an approximation of an “identity” function.

• It represents original input from compressed data.

• It consists of narrow hidden layer between Encoder & Decoder


<p align="center">
  <img alt="For-the-Badge-Python" src="https://www.jeremyjordan.me/content/images/2018/03/Screen-Shot-2018-03-07-at-8.24.37-AM.png">
  
</p>





Autoencoder neural network is an unsupervised learning algorithm that applies backpropagation,
setting the target values to be equal to the inputs.
To do so it converts the input into a latent space of reduced dimensions using a encoder part and
then again reconstructs it using decoder , the result of this whole cycle of construction and
reconstruction consist of a single epoch . It is trained over multiple such epochs and it gets better
over time .
The output is evaluated using a defined loss function which over time predicts how good is the
reconstruction of the input and with every epoch it tries to reduce the loss and improves accuracy
over time . 

We can twist these networks a little and take advantages of the reduced latent space to represent
the reduced dimensions and train these networks over multiple images of the desired output of
the same image to train the network in the way we can extract the desired results from the dataset



 Like in the noise reduction model we will label our input dataset as a noisy image in the working directory and the output labels to clean images and train the model to generate noiseless images and reducing the loss function over it .
 

 In the super resolution we will blur the images in the input directory and then train them to get the clear version as output directory labels and train the loss function over it . Thus when we test on a clear picture it will increase its resolution .



The dimensionality reduction works on the basic working principal of AE when the state is reduced to latent space for representation we extract the latent space as it can represent the data efficiently . Generally we use PCA for it But AE perform better them PCA in almost all the given conditions . 




### Content :

 * [Noise Reduction using AutoEncoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Noise-Reduction-using-Autoencoders)
 > Training a autoencoder to reduce noise in the given image .
 
 * [Image Super resolution using AutoEncoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Image-SuperResolution-using-AutoEncoders)
 > Training a autoencoder to increase the resolution of a image .
 
 * [Dimensionality Reduction using Autoencoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Dimensionality-Reduction-using-AutoEncoders)

 > Training a autoencoder and extracting the reduced dimensions from the bottleneck to reduce the dimensions required to represent and visualise data .


<p align="center">
  <img alt="For-the-Badge-Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg">
  
</p>
