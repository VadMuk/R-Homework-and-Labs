# **Homework  11**

# **1.Goal**

Implement a classification of iris plant varieties (Iris Setosa - 0, Iris Versicolor - 1, Iris Virginica - 2) based on four characteristics: the size of the pistils and stamens of its flowers.

# **2.Tasks**

1.Download dataset

2. Create an ANN model in Keras
   
3. Set up training parameters
   
4. Train and evaluate the model

# **3.Execution of work**

Multi-class classification is one of the main types of problems solved by neural networks. Listing 1 shows an example of the data.


Listing 1 - Sample Data

<pre> 
5.1,3.5,1.4,0.2, Iris-setosa

4.9,3.0,1.4,0.2, Iris-setosa

4.7,3.2,1.3,0.2, Iris-setosa

4.6,3.1,1.5,0.2, Iris-setosa

5.0,3.6,1.4,0.2,Iris-setosa  </pre>

The dataset is available for download.follow the link(  the iris.data file). The downloaded file must be renamed to “iris.csv” and placed in your project directory.
We import the necessary classes and functions. In addition to Keras, we'll need Pandas for loading data and scikit-learn for data preparation and model evaluation (Listing 2).

Listing 2 - Connecting modules

<pre>
import pandas

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder   </pre>

The dataset is loaded directly using pandas. Then, the attributes (columns) need to be split into input (X) and output (Y) (Listing 3).

Listing 3 - Loading data
<pre>

dataframe = pandas.read_csv("iris.csv", header=None)

dataset = dataframe.values

X = dataset[:,0:4].astype(float)

Y = dataset[:,4]
</pre>
When solving multi-class classification problems, it is good practice to transform the output attributes from a vector to a matrix, as shown in Listing 4.

Listing 4 - Data Representation

Iris-setosa, Iris-versicolor, Iris-virginica

<pre>
1, 0, 0

0, 1, 0

0, 0, 1 </pre>

To do this you need to use the function to_categorical()(Listing 5)

Listing 5 - Transforming from text labels to a categorical vector

<pre>
encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

dummy_y = to_categorical(encoded_Y)
</pre>
Now we can define the basic network architecture (Listing 6)

Listing 6 - Creating a Model
<pre>
model = Sequential()

model.add(Dense(4, activation='relu'))

model.add(Dense(3, activation='softmax'))
</pre>

The fundamental building block of neural networks is the layer, a data processing module that can be thought of as a filter for the data. It takes in some data and outputs it in a more useful form. 

Specifically, layers extract representations from the data fed to them, which hopefully will be more meaningful for the task at hand. In essence, deep learning is a technique that combines simple layers that implement some form of stepwise data purification. A deep learning model can be compared to a sieve consisting of a sequence of increasingly fine-grained data filters—layers.

In this case, our network consists of a sequence of two dense layers, which are tightly connected (also called fully connected) neural layers. 

The second (and final) layer is a 3-variable softmax loss layer, returning an array of 3 probability estimates (summing to 1). 

Each estimate determines the probability that the current image belongs to one of 3 color classes.

To prepare the network for training, you need to configure three more parameters for the compilation stage:

1)a loss function that determines how the network should evaluate the quality of its performance on training data and, accordingly, how to adjust it in the right direction;

2)optimizer - a mechanism by which the network will update itself based on observed data and a loss function;

3)metrics for monitoring during the training and testing stages - here we will only be interested in accuracy (the proportion of correctly classified images).


Listing 6 - Initializing training parameters

<pre> model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) </pre>

Now you can start training the network (Listing 7), for which, if you are using the Keras library, you just need to call the network's fit method - it tries to adapt (fit) the model to the training data.


Listing 7 - Network training

<pre> model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1) </pre>

During the training process, four quantities are displayed: the network loss on the training data and the network accuracy on the training data, as well as the loss and accuracy on the data that was not involved in the training.

# **4.Requirements**

1. Explore different ANN architectures: different number of layers, different number of neurons per layer
   
2. Study learning under different learning parameters (function parameters fit)
   
3. Plot error and accuracy graphs during training
   
4. Choose the best model
