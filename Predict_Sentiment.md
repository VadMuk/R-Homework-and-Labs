
# Target of Project2
Predict Sentiment From Movie Reviews

# Tasks
To study the methods of representing text for transmission to the ANN
Achieve a forecast accuracy of at least 95%

# Execution of work
What is sentiment analysis?
Sentiment analysis can be used to determine a person's attitude (e.g., mood) toward a text, interaction, or event. Therefore, sentiment analysis belongs to the field of natural language processing, in which the meaning of a text must be deciphered to extract sentiment and mood.

The sentiment spectrum is typically divided into positive, negative, and neutral categories. Using sentiment analysis, one can, for example, predict customer opinions and attitudes toward a product based on the reviews they write. Therefore, sentiment analysis is widely applied to reviews, surveys, texts, and much more.


# IMDb dataset

The IMDb dataset consists of 50,000 user-generated movie reviews, categorized as positive (1) and negative (0).
The reviews are pre-processed and each is encoded as a sequence of word indices as integers.
Words in reviews are indexed by their overall frequency in the dataset. For example, the integer "2" encodes the second most frequently used word.
50,000 reviews are divided into two sets: 25,000 for training and 25,000 for testing.
The dataset was created by Stanford University researchers and presented in a 2011 paper, which achieved 88.89% prediction accuracy. The dataset was also used in the 2011 Kaggle community competition "Bag of Words Meets Bags of Popcorn."

# Importing dependencies and retrieving data
Let's start by importing the necessary dependencies for data preprocessing and model building.
<pre></pre>
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
</pre>

Let's load the IMDb dataset, which is already built into Keras. Since we don't want a 50/50 split of training and testing data, we'll immediately merge the data after loading for an 80/20 split:

<pre>
from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
</pre>
  
# Data mining

Let's explore our dataset:
<pre>
print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))
Categories: [0 1]
Number of unique words: 9998
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))
Average Review length: 234.75892
Standard Deviation: 173.0

</pre>
You can see that all data falls into two categories: 0 or 1, representing the sentiment of the review. The entire dataset contains 9998 unique words, with the average review length being 234 words with a standard deviation of 173.
Let's look at a simple way of learning:

</pre>

<pre>
print("Label:", targets[0])
Label: 1
print(data[0])
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
</pre>

Here you see the first review from the dataset, which is marked as positive (1). The code below converts the indexes back to words so we can read them. It replaces each unknown word with "#." This is done using the get_word_index() function.

<pre>
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
print(decoded)
this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert # is an amazing actor and now the same being director # father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for # and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and definitely this was also # to the two little boys that played the # of Norman and paul they were just brilliant children are often left out of the # list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
</pre>

# Data preparation
It's time to prepare the data. We need to vectorize each review and fill it with zeros so that the vector contains exactly 10,000 numbers. This means we fill every review shorter than 10,000 words with zeros. This is because the largest review is almost the same size, and each input element of our neural network should be the same size. We also need to convert the variables to floats.

<pre>
def vectorize(sequences, dimension = 10000):
results = np.zeros((len(sequences), dimension))
for i, sequence in enumerate(sequences):
results[i, sequence] = 1
return results
data = vectorize(data)
targets = np.array(targets).astype("float32")
We'll split the dataset into training and testing sets. The training set will consist of 40,000 reviews, and the testing set will consist of 10,000.
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
</pre>

# Model creation and training
Now we can create a simple neural network. Let's start by defining the type of model we want to create. Keras offers two types of models: sequential and those with a functional API.
Next, we need to add the input, hidden, and output layers. To prevent overfitting, we'll use dropout between them. Note that you should always use a dropout rate between 20% and 50%. Each layer uses the "dense" function to fully connect the layers together. In the hidden layers, we'll use the "relu" activation function, as this almost always produces satisfactory results. Feel free to experiment with other activation functions. For the output layer, we'll use a sigmoid function, which renormalizes values ​​between 0 and 1. Note that we're setting the input feature size to 10,000 because our reviews are up to 10,000 integers in size. The input layer accepts features of size 10,000 and outputs features of size 50.
Finally, let Keras print a short description of the model we just created.
<pre>
Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu")
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
Output Layer
model.add(layers.Dense(1, activation = "sigmoid"))model.summary()
model.summary()
</pre>

<pre>
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
dense_1 (Dense) (None, 50) 500050
_________________________________________________________________
dropout_1 (Dropout) (None, 50) 0
_________________________________________________________________
dense_2 (Dense) (None, 50) 2550
_________________________________________________________________
dropout_2 (Dropout) (None, 50) 0
_________________________________________________________________
dense_3 (Dense) (None, 50) 2550
_________________________________________________________________
dense_4 (Dense) (None, 1) 51
=================================================================
Total params: 505,201
Trainable params: 505,201
Non-trainable params: 0
_________________________________________________________________
</pre>
  
  Now we need to compile our model, essentially setting it up for training. We'll use the "adam" optimizer. An optimizer is an algorithm that modifies weights and biases during training. We'll use binary cross-entropy as the loss function (since we're working with binary classification), and accuracy as the evaluation metric.

<pre>
model.compile(
optimizer = "adam",
loss = "binary_crossentropy",
metrics = ["accuracy"]
)
</pre>

Now we can train our model. We'll do this with a batch size of 500 and only two epochs, as I've found that the model starts to overfit the longer it trains, the more likely it is to overfit. The batch size determines the number of samples propagated through the network, and an epoch is a single pass through all the samples in the dataset. A larger batch size typically leads to faster training, but not always faster convergence. A smaller batch size trains more slowly but can converge faster. The choice of one or the other definitely depends on the type of problem you're solving, and it's best to try each one. If you're new to this, I'd recommend starting with a batch size of 32, which is somewhat of a standard.
<pre>
results = model.fit(
train_x, train_y,
epochs= 2,
batch_size = 500,
validation_data = (test_x, test_y)
)
</pre>

<pre>
Train on 40000 samples, validate on 10000 samples
Epoch 1/2
40000/40000 [=============================] - 5s 129us/step - loss: 0.4051 - acc: 0.8212 - val_loss: 0.2635 - val_acc: 0.8945
Epoch 2/2
40000/40000 [=============================] - 4s 90us/step - loss: 0.2122 - acc: 0.9190 - val_loss: 0.2598 - val_acc: 0.8950
</pre>

Let's evaluate the model's performance:
<pre>
print(np.mean(results.history["val_acc"]))
0.894750000536
</pre>
  
# Requirements
.Build and train a neural network for text processing

.To investigate the results for different sizes of text representation vector

.Write a function that allows you to enter custom text (in the report, provide an example of how the network works with custom text)
