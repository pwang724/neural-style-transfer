# Neural Style Transfer

Simple tensorflow implementation of NST by [Gatys and Bethge](https://arxiv.org/abs/1508.06576)

Implementation of the original algorithm uses a pre-trained [VGG19 network](https://github.com/machrisaa/tensorflow-vgg). As described in paper, 

Content layer = **[conv4~2~]** 
Style layer = **[conv1~1~, conv2~1~, conv3~1~, conv4~1~, and conv5~1~]**. 
Style weight = **[.2, .2, .2, .2, .2]**

Style is defined as a correlation matrix, where the **_ij_**th value of the matrix is the correlation value between the filtered images at depth **i** and depth **j** at a particular layer.

<div>
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/jade_selfie.jpg" width="200">
</div>

Neural dream: how can I get a network to reveal, through its connectivity patterns, the data that has shaped it?

1. 
GOAL: have input and target image. want to reveal the distance between target and input images based on the objects that it has seen. i.e. define 'image' distance not to be euclidian but based on prior basis functions.

TODO: take two images, input and target. start from input image, and perform gradient descent with loss being MSE(target-input) on a specific layer. see what intermediate output images look like. 

2.

GOAL: NST works because style and content are described in terms of frequency (filter correlation) and position (pixel intensities). these two quantities are maximally independent. therefore, the goal of maximizing both quantities at once can lead to unconstrained solutions. what happens when certain quantities are more constrained (i.e. layer 1 vs layer 5 activation)?

TODO: have input and target image. hold still one layer of input and another layer of input. loss is mse(input layer 1 - loss layer 1) + mse(output layer 5 - input layer 5). 

TODO: do the same except with further downstream layers, such as a feed forward layer


3. deepdream is simply holding one layer constant and backprop'ing. similar to #1. 

https://arxiv.org/pdf/1312.6034v2.pdf
https://arxiv.org/pdf/1506.06579.pdf
https://arxiv.org/pdf/1409.4842.pdf

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
