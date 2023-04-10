import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip, pickle
import time
import os

# Load the pre-pared data file (contains two 1D arrays of 50 points each:
# 50 inputs stored in variable x_train and 50 target outputs stored in variable
# y_hat_train.
with gzip.open(filename=os.path.join("..","data","dataset1.pgz")) as f:
    x_train, y_hat_train = pickle.load(f)


# Create a matplotlib figure
fh = plt.figure(figsize=(8,4))
# Dividing the figure into 1 x 2 plots, create the first one (starting from the top-left)
ph = fh.add_subplot(1,2,1)
# Plot y_hat_train over x_train as blue points
ph.scatter(x=x_train, y=y_hat_train, color='tab:blue',label='Train')
# Add axis labels
ph.set_xlim(0,5)
ph.set_ylim(0,6)
ph.set_xlabel('x')
ph.set_ylabel('y')

# Show figure in interactive mode
# (tnat is, do not stop the execution of the script)
# when calling plt.show()
plt.ion()
# Delay a bit to give matplotlib a chance to finish drawing to screen
plt.pause(0.01)
time.sleep(1)
# Show the figure (it won't block the program execution because interactive
# mode is on
plt.show()

# The objective is to create the network that models the relationship between
# value in x_train and corresponding target value in y_hat_train.

# Create a new neural network model
net = tf.keras.models.Sequential()

# Create a fully connected layer of 128 sigmoid units (specify the size
# of the input, which in this case is 1)
layer = tf.keras.layers.Dense(units=128, input_shape= (1,),activation='sigmoid')
# Add the layer to the network
net.add(layer=layer)

# Create another  fully connected layer of 64 sigmoid units (don't need to specify the
# input size - it will be inferred from the number of units in the previous layer)
layer = tf.keras.layers.Dense(units=16,activation='sigmoid')
# Add the layer to the network
net.add(layer=layer)

# Create the final fully connected layer, with just one linear neuron that
# will be the output of the neural network
layer = tf.keras.layers.Dense(units=1, activation='linear')
# Add the layer to the network
net.add(layer=layer)

# Compile the network model - this is where weight matrices and
# bias vectors for each layer will get instantiated and initialised
# to random values.  Compiling also needs to know about the loss function that
# you want to use for training (in this case MSE loss) and the type of optimiser (Adam optimiser in this case).
net.compile(optimizer='adam', loss='mse')

# Train the network on the 50 single-value inputs and the corresponding single-value
# target values; train over 500 epochs - that is going through all the data 500 times,
# each time adjusting the weights and biases to improve the model (to better predict
# y_hat_train from x_train).
# The output of the fit method provides various information about the training; train_info.history
# is a dictionary of various things tracked throughout the training; train_info.history['loss']
# gives the loss evaluation over each epoch (in this case the 'mse' loss over the training 500 epochs).
train_info = net.fit(x=x_train, y=y_hat_train, epochs=500)

# Create a test set - 50 new values evenly spaced out in the range from 0 to 5
x_test = np.linspace(start=0,stop=5,num=100)
# Compute the output of the network for those 100 test values
y_test = net.predict(x=x_test)

# Add the predicted output of the networks as a continuous line
ph.plot(x_test, y_test, color='tab:orange',label='Test')
plt.legend() #Show the plot legend
plt.pause(0.01)
time.sleep(1)
# Delay a bit to give matplotlib a chance to finish drawing to screen
plt.show()


# Add second plot to the 1x2 figure
ph = fh.add_subplot(1,2,2)
# ...the values of loss over epochs
ph.plot(train_info.history['loss'],color='tab:green')
ph.set_xlabel('Epoch')
ph.set_ylabel('MSE')
ph.set_title('Training history')

# Disable interactive plot mode
plt.ioff()
# Now, calling plot show how plot block the execution of the script until the figure is closed
plt.show()

# Other useful methods of the Sequential model (referenced by the 'net' variable
# in this example)
#
# net.save('filename') - saves the model to file name (next time it can be loaded
#                         and used without training)
#
# net = tf.keras.models.load_model('filename') - load the previously saved model
#
#
# W, b=net.layers[l].get_weights() - fetches the values of weights and biases
#                                 from layer l, where l is index from 0 to L-1,
#                                 where L is the total number of layers.  W
#                                 will be a matrix of weights and b a vector
#                                 of biases.
#
# net.layers[l].set_weights([W,b]) - sets the values of weights and biases
#                                    of layer l (index from 0 to L-1).  W must
#                                    be a matrix of the size IxU where I is
#                                    the number of inputs (or units of previous
#                                    layer and U is the number of units of
#                                    layer l; b must be a vector of U bias
#                                    values.