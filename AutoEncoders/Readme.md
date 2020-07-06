# Autoencoders

Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning.
Specifically, we'll design a neural network architecture such that we impose a bottleneck in the network which forces a compressed knowledge representation of the original input. 
If the input features were each independent of one another, this compression and subsequent reconstruction would be a very difficult task. 
However, if some sort of structure exists in the data (ie. correlations between input features), this structure can be learned and consequently leveraged when 
forcing the input through the network's bottleneck.

## Autoencoder Components:
#### Autoencoders consists of 4 main parts:
1- Encoder: In which the model learns how to reduce the input dimensions and compress the input data into an encoded representation.
2- Bottleneck: which is the layer that contains the compressed representation of the input data. This is the lowest possible dimensions of the input data.
3- Decoder: In which the model learns how to reconstruct the data from the encoded representation to be as close to the original input as possible.
4- Reconstruction Loss: This is the method that measures measure how well the decoder is performing and how close the output is to the original input.
The training then involves using back propagation in order to minimize the network’s reconstruction loss.

