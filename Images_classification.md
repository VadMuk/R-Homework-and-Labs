
# Target
Implement classification of black-and-white images of handwritten digits (28x28) into 10 categories (from 0 to 9).


<img width="692" height="519" alt="image" src="https://github.com/user-attachments/assets/7e4743b5-2c28-409a-ba4d-ccf91f8be4d8" />


The dataset contains 60,000 training images and 10,000 testing images.
More details: 
https://keras.io/datasets/ http://yann.lecun.com/exdb/mnist/

# Tasks
- Explore the graphical data representation
- Learn the simplest way to transfer graphical data to a neural network
- Create a model
- Set up training parameters
- Write a function that allows users to upload an image and classify it

# Execution of work
The MNIST dataset is already included in Keras in the form of a set of four Numpy arrays.

Listing 1:
<pre>
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
</pre>

Here, train_images and train_labels are the training set, i.e., the data needed for training. After training, the model will be tested on the test (or validation) set, test_images and test_labels. The images are stored in Numpy arrays, and the labels are in an array of numbers from 0 to 9. There is a one-to-one correspondence between the images and labels.
To check if the download was correct, simply compare the test image with its label.

Listing 2:
<pre>
import matplotlib.pyplot as plt
plt.imshow(train_images[0],cmap=plt.cm.binary)
plt.show()
print(train_labels[0])
</pre>
The original images are represented as arrays of numbers in the interval [0, 255]. Before training, they must be transformed so that all values ​​are in the interval [0, 1].

Listing 3:
<pre>
train_images = train_images / 255.0
test_images = test_images / 255.0
</pre>
Category labels also need to be encoded. In this case, direct encoding of labels involves constructing a vector of zero-based elements with the value 1 at the element whose index corresponds to the label index.

Listing 4:
<pre>
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
</pre>

Now you can set up the basic network architecture.

Listing 5:
<pre>
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
</pre>

To prepare the network for training, you need to configure three more parameters for the compilation stage:
.a loss function that determines how the network should evaluate the quality of its performance on training data and, accordingly, how to adjust it in the right direction;
.optimizer - a mechanism by which the network will update itself based on observed data and a loss function;
.metrics for monitoring during the training and testing stages - here we will only be interested in accuracy (the proportion of correctly classified images).

Listing 6:
<pre>
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
</pre>
Now you can start training the network. If you are using the Keras library, you can simply call the network's fit method—it attempts to adapt (fit) the model to the training data.

Listing 7:
<pre>
model.fit(train_images, train_labels, epochs=5, batch_size=128)
</pre>
During the training process, two quantities are displayed: the network loss on the training data and the network accuracy on the training data.
Now let's check how the model recognizes the test set:

Listing 8:
<pre>
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
</pre>
  
# Requirements

.Find a network architecture that will achieve classification accuracy of at least 95%

.To investigate the influence of different optimizers and their parameters on the learning process

.Write a function that will allow loading a custom image not from a dataset
