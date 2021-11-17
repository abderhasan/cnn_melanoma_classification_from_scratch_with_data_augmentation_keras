# Training a CNN from scratch with data augmentation for melanoma detection using Keras

I will be sharing a script using Keras for training a Convolutional Neural Network (CNN) from scratch with data augmentation for melanoma detection. You can view the code from <a href="https://github.com/abderhasan/cnn_melanoma_classification_from_scratch_with_data_augmentation_keras"><strong>this GitHub repository</strong></a>.

Before proceeding, make sure that you structure the data as follows (the numbers represent the number of images in each file):

![alt text](https://github.com/abderhasan/cnn_melanoma_classification_from_scratch_keras/blob/master/directory-structure.png)

You can download the data from,<strong> <a href="https://drive.google.com/drive/folders/126UgFt_xqnHpeV1Pr_qLDQLwzBbi4rrY?usp=sharing">here</a></strong>. I used two classes as you can see from the figure above (nevus and melanoma). For training, I kept 374 images in each class to keep the data balanced.

To run the code:

`$ python cnn_with_augmentation.py`

The results will not be optimal, as the purpose is to show how one can train a CNN from scratch.

<strong>What variables to edit in the code?</strong>

You need to edit the following variables to point to your data:

<em>train_directory</em> (path to your training directory)

<em>validation_directory</em> (path to your training directory)

<em>test_directory</em> (path to your testing directory)
