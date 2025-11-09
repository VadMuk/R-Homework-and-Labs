# Homework 13

# Target
To predict the median price of homes in the Boston suburbs in the mid-1970s using data such as crime rates, local property tax rates, etc.

This dataset contains relatively few data samples: only 506, divided into 404 training and 102 validation samples. Each feature in the input data (e.g., crime rate) has its own scale. For example, some features are proportions and have values ​​between 0 and 1, others between 1 and 12, and so on.

# Tasks
1. Create a model
   
2. Set up training parameters

3. Train and evaluate models

4. Explore cross-validation

# Execution of work
The dataset is included with Keras.

Listing 1:
<pre> 
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
   (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)  </pre>

404 training and 102 test samples, each with 13 numerical features.
Prices generally range from $10,000 to $50,000.

Feeding values ​​with widely varying ranges into a neural network would be problematic. The network could, of course, automatically adapt to such heterogeneous data, but this would complicate training. 

In practice, normalization is commonly applied to such data: for each feature in the input data (column in the input matrix), the mean for that feature is subtracted from each value, and the difference is divided by the standard deviation. As a result, the feature is centered at zero and has a standard deviation of one. This normalization is easily performed using Numpy.

Listing 2:
<pre>
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
</pre>
Let's define the build_model() function:

Listing 3:
<pre>
def build_model():
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
return model
</pre>
The network ends with a one-dimensional layer without an activation function (a linear layer). This is a typical configuration for scalar regression (which aims to predict a single value on a continuous number line). Applying an activation function could limit the range of output values: for example, if a sigmoid activation function were applied in the final layer, the network would learn to predict only values between 0 and 1.
In this case, with a linear final layer, the network is able to predict values ​​from any range.

Note that the network is compiled with the mse (mean squared error) loss function, which calculates the squared difference between the predicted and target values. This function is widely used in regression problems. A new parameter has also been added to the training phase: mae (mean absolute error). This is the absolute value of the difference between the predicted and target values. For example, an MAE value of 0.5 in this problem means that, on average, predictions are off by $500.

To assess the network's quality while adjusting its parameters (such as the number of training epochs), we can split the source data into training and validation sets, as we did in the previous examples. However, since our dataset is already small, the validation set would be too small (say, around 100 samples). As a result, validation scores can vary significantly depending on the data included in the validation and training sets: validation scores may have too wide a spread, making it difficult to reliably assess the model's quality.
A good practice in such situations is to use K-fold cross-validation. It involves dividing the available data into K bins (usually K = 4 or 5), creating K identical models, and training each on K - 1 bins, with evaluations based on the remaining bins. The resulting K evaluations are then averaged and used as the model score. Implementing this type of cross-validation in code is quite simple.

Listing 4:
<pre>
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
print('processing fold #', i)
val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
axis=0)
partial_train_targets = np.concatenate(
[train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
model = build_model()
model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
all_scores.append(val_mae)
print(np.mean(all_scores))
</pre>
Different runs indeed show different estimates, ranging from 2.6 to 3.2. The average (3.0) appears more reliable than any of the individual runs' estimates—this is the main value of K-block cross-validation. 

In this case, the average error was $3,000, which is quite significant considering that prices range from $10,000 to $50,000.

It is necessary to reduce or increase the number of training epochs and analyze the obtained results.

# Requirements
1) Explain the differences between classification and regression problems
2) To study the influence of the number of epochs on the model training result
3) Identify the point of retraining
4) Apply K-block cross-validation for different K
5) Plot error and accuracy plots during training for models, as well as average plots across all models
