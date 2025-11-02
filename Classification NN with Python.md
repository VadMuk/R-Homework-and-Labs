# **Laboratory work 3

# **Goal
Implement classification between rocks (R) and metal cylinders (M) based on radar signal reflection data from surfaces.
The 60 input values ​​represent the reflected signal strength at a given angle. The input data are normalized and range from 0 to 1.
Tasks
Get acquainted with the binary classification problem
Download data
Create an ANN model in tf.Keras
Set up training parameters
Train and evaluate the model
Modify the model and conduct a comparison. Explain the results.
Execution of work
Below are the first 2 rows from the data set.
Listing 1:
0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857, 0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641, R
0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028, 0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621, R
The dataset is available for download.follow the link(sonar.all-data file). The downloaded file must be renamed to "sonar.csv" and placed in your project directory.
We'll import the necessary classes and functions. In addition to Keras, we'll need Pandas for loading data and scikit-learn for data preparation and model evaluation.
Listing 2:
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
The dataset is loaded directly using pandas. Then, the attributes (columns) need to be split into 60 input parameters (X) and 1 output parameter (Y).
Listing 4:
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
The output parameters are represented by strings ("R" and "M"), which must be converted to integer values ​​0 and 1, respectively. For this purpose, the LabelEncoder from scikit-learn is used.
Listing 5:
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Now you can set up the basic network architecture.
Listing 6:
model = Sequential()
model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))
To prepare the network for training, you need to configure three more parameters for the compilation stage:
.A loss function that determines how the network should evaluate the quality of its performance on training data and, accordingly, how to adjust it in the right direction; For binary classification problems, the binary crossentropy function is used.
.optimizer - a mechanism by which the network will update itself based on observed data and a loss function;
.metrics for monitoring during the training and testing stages - here we will only be interested in accuracy (the proportion of correctly classified images).
Listing 7:
model.compile(optimizer='adam',loss='binary crossentropy', metrics=['accuracy'])
Now you can start training the network. If you are using the Keras library, you can simply call the network's fit method—it attempts to adapt (fit) the model to the training data.
Listing 8:
model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)
During the training process, four quantities are displayed: the network loss on the training data and the network accuracy on the training data, as well as the loss and accuracy on the data that was not involved in the training.
The presented dataset contains some redundancy, as the same signal is described from different angles. It is likely that some signal reflection angles are more significant than others. Changing the number of neurons in the input layer directly affects the number of features the neural network will process.
It is necessary to reduce the size of the input layer by half and compare with the results of the original architecture.
A neural network with multiple layers allows for finding patterns not only in the input data but also in its combination. Additional layers also allow for nonlinearity to be introduced into the network, resulting in higher accuracy.
It is necessary to add an intermediate (hidden) Dense layer to the network architecture with 15 neurons and analyze the results.

Requirements
.To study the influence of the number of neurons in a layer on the model training result.
.To study the influence of the number of layers on the model training result
.Plot error and accuracy graphs during training
.Conduct a comparison of the obtained networks and explain the results
