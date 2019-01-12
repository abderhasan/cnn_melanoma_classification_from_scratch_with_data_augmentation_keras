# Training a CNN from scratch with data augmentation for melanoma detection using Keras

I will be sharing a script using Keras for training a Convolutional Neural Network (CNN) from scratch with data augmentation for melanoma detection. You can view the code from <a href="https://github.com/abderhasan/cnn_melanoma_classification_from_scratch_with_data_augmentation_keras"><strong>this GitHub repository</strong></a>. In the <a href="https://abder.io/training-a-cnn-from-scratch-for-melanoma-detection-using-keras/"><strong>previous post</strong></a>, the CNN was trained from scratch without augmenting the data.

Before proceeding, make sure that you structure the data as follows (the numbers represent the number of images in each file):

![alt text](https://github.com/abderhasan/cnn_melanoma_classification_from_scratch_keras/blob/master/directory-structure.png)

You can download the data from,<strong> <a href="https://drive.google.com/drive/folders/126UgFt_xqnHpeV1Pr_qLDQLwzBbi4rrY?usp=sharing">here</a></strong>. I used two classes as you can see from the figure above (nevus and melanoma). For training, I kept 374 images in each class to keep the data balanced.

The results will not be optimal, as the purpose is to show how one can train a CNN from scratch.

<strong>What variables to edit in the code?</strong>

You need to edit the following variables to point to your data:

<em>train_directory</em> (path to your training directory)

<em>validation_directory</em> (path to your training directory)

<em>test_directory</em> (path to your testing directory)

<strong>What should you expect (outputs)?</strong>

<em>Training and validation accuracy</em>

<img class="aligncenter size-full wp-image-863" src="https://abder.io/wp-content/uploads/2019/01/accuracy-1.png" alt="" width="640" height="480" />

<em>Training and validation loss</em>

<img class="aligncenter size-full wp-image-864" src="https://abder.io/wp-content/uploads/2019/01/loss-1.png" alt="" width="640" height="480" />

<em>ROC curve</em>

<img class="aligncenter size-full wp-image-865" src="https://abder.io/wp-content/uploads/2019/01/ROC-1.png" alt="" width="640" height="480" />

In addition to some other values (i.e. accuracy, confusion matrix) that will be displayed on the console.
