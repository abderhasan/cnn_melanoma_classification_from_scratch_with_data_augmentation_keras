'''
This code enables you to train a Convolutional Neural Network (CNN) with data augmentation
Abder-Rahman Ali
abder@cs.stir.ac.uk
'''

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# path to the training, validation, and testing directories
train_directory = '/train'
validation_directory = '/valid'
test_directory = '/test'
correct_classification = 0
number_of_test_images = 0
labels = []
prediction_probabilities = []
predictions = []


# define the Convolutional Neural Network (CNN) model
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(512,512,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(512,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

# preprocess the data
train_data = ImageDataGenerator(featurewise_center=True,
	    featurewise_std_normalization=True,
	    rotation_range=180,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True)
validation_data = ImageDataGenerator()

train_generator = train_data.flow_from_directory(train_directory,target_size=(512,512),batch_size=20,class_mode='binary')
validation_generator = validation_data.flow_from_directory(validation_directory,target_size=(512,512),batch_size=20,class_mode='binary')

# fit the model to the data
history = model.fit_generator(train_generator,
	steps_per_epoch=50,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=5)

# save the model
model.save('benign_and_melanoma_from_scratch_with_augmentation.h5')

# generate accuracy and loss curves for the training process (history of accuracy and loss)
accuracy = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

number_of_epochs = range(1,len(accuracy)+1)

plt.plot(number_of_epochs, accuracy, 'rx', label='Training accuracy')
plt.plot(number_of_epochs, val_acc, 'bx', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy.png')

plt.close()

plt.plot(number_of_epochs, loss, 'ro', label='Training loss')
plt.plot(number_of_epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')

plt.close()

# test the model
for root, dirs, files in os.walk(test_directory):
	for file in files:
		img = cv2.imread(root + '/' + file)
		img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
		img = np.expand_dims(img,axis=0)
		if os.path.basename(root) == 'nevus':
			label = 0
		elif os.path.basename(root) == 'melanoma':
			label = 1
		labels.append(label)
		img_class = model.predict_classes(img)
		img_class_probability = model.predict_proba(img)
		prediction_probability = img_class_probability[0]
		print 'This is the prediction probability'
		print prediction_probability
		prediction_probabilities.append(prediction_probability)
		prediction = img_class[0]
		if prediction == label:
			correct_classification = correct_classification + 1
		predictions.append(prediction)
		print 'This is the prediction: '
		print prediction
		number_of_test_images = number_of_test_images + 1

print 'number of correct results:'
print correct_classification

print 'number of test images'
print number_of_test_images

print 'Accuray:'
print str((float(correct_classification)/float(number_of_test_images)) * 100) + '%'

fpr, tpr, thresholds = roc_curve(labels, prediction_probabilities)
auc_value = auc(fpr, tpr)

print 'This is the AUC value'
print auc_value

# plot the ROC curve
plt.plot(fpr, tpr, label='CNN (area = {:.3f})'.format(auc_value))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('ROC.png')

plt.close()

# create the confusion matrix
cm = confusion_matrix(labels,predictions)
print cm
