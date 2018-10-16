# Neural Style Transfer

Simple tensorflow implementation of NST by [Gatys and Bethge](https://arxiv.org/abs/1508.06576). Synthesizes a composite image with content specified by one image and style specified by another.

Implementation of the original algorithm uses a pre-trained [VGG19 network](https://github.com/machrisaa/tensorflow-vgg), with 

Content layer = **[conv4_2]** 
Style layer = **[conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1]**. 
Style weight = **[.2, .2, .2, .2, .2]**

"Style" is defined as a correlation matrix, where the **_ij_** th value of the matrix is dot product between the filtered images at depth **i** and depth **j** at a particular layer. Style loss is defined as the MSE between the correlation matrices of the input image and style image for a particular layer. This loss is summed and weighted for multiple layers as specified above. "Content" is defined as the activations of a particular layer. Content loss is defined as the MSE between the activations of the input image and the content image for a particular layer. Total loss is defined as the weighted sum of style loss and content loss.

##### Example
<div>
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/jade_selfie.jpg" width="200">
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/jade.jpg" width="200">
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/out_3000.jpg" width="200">
</div>

##### Command

```
python test.py --content content_dir --style style_dir --model vgg19_dir --lr 1 --epoch 4000 --alpha 1 --beta 10000
```

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
