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
  <img alt="For-the-Badge-Python" src="data:image/png;base64,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">
  
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





 * [Noise Reduction using AutoEncoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Noise-Reduction-using-Autoencoders)
 > Training a autoencoder to reduce noise in the given image .
 
 * [Image Super resolution using AutoEncoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Image-SuperResolution-using-AutoEncoders)
 > Training a autoencoder to increase the resolution of a image .
 
 * [Dimensionality Reduction using Autoencoders](http://github.com/Storiesbyharshit/Data-Science-Portfolio/tree/master/AutoEncoders/Dimensionality-Reduction-using-AutoEncoders)

 > Training a autoencoder and extracting the reduced dimensions from the bottleneck to reduce the dimensions required to represent and visualise data .


<p align="center">
  <img alt="For-the-Badge-Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg">
  
</p>
