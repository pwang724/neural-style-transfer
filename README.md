# Neural Style Transfer

Simple tensorflow implementation of NST by [Gatys and Bethge](https://arxiv.org/abs/1508.06576). Synthesizes a composite image with content specified by one image and style specified by another.

Implementation of the original algorithm uses a pre-trained [VGG19 network](https://github.com/machrisaa/tensorflow-vgg), with

Content layer = **[conv4_2]**\
Style layer = **[conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1]**\
Style weight = **[.2, .2, .2, .2, .2]**

"Style" is defined as a correlation matrix, where the **_ij_** th value of the matrix is dot product between the filtered images at depth **i** and depth **j** at a particular layer. Style loss is defined as the MSE between the correlation matrices of the input and style image at a particular layer.

"Content" is defined as the activations of a particular layer. Content loss is defined as the MSE between the activations of the input and content image for a particular layer. Total loss is defined as the weighted sum of style loss and content loss.

#### Example
<div>
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/jade_selfie.jpg" width="200">
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/jade.jpg" width="200">
<img src="https://raw.githubusercontent.com/pwang724/neural-style-transfer/master/example/out_3000.jpg" width="200">
</div>

##### Command

```
python stylize.py --content content_dir --style style_dir --model vgg19_dir --lr 1 --epoch 4000
```