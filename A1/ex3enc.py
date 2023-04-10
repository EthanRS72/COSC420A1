import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
import os
import pickle, gzip

load_from_file = True

# Load the CIFAR10 dataset
data = tf.keras.datasets.cifar10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_hat_train), (x_test, y_hat_test) = data.load_data()

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'cifar10enc')
net_save_name = save_name + '_cnn_net.h5'
history_save_name = save_name + '_cnn_net.hist'

# Show 16 train images with the corresponding labels
show_methods.show_data_images(images=x_train[:16],labels=y_hat_train[:16],class_names=class_names,blocking=False)

n_classes = len(class_names)

_,H,W,C = np.shape(x_train)

if load_from_file and os.path.isfile(net_save_name):
   # ***************************************************
   # * Loading previously trained neural network model *
   # ***************************************************

   # Load the model from file
   print("Loading neural network from %s..." % net_save_name)
   net = tf.keras.models.load_model(net_save_name)

   # Load the training history - since it should have been created right after
   # saving the model
   if os.path.isfile(history_save_name):
      with gzip.open(history_save_name) as f:
         history = pickle.load(f)
   else:
      history = []
else:
   # ************************************************
   # * Creating and training a neural network model *
   # ************************************************

   # Create feed-forward network
   net = tf.keras.models.Sequential()

   # Add a convolutional layer, 3x3 window, 64 filters - specify the size of the input as 32x32x3, padding="same"
   net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu', input_shape=(H, W, C),padding='same'))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

   # Add a convolutional layer, 3x3 window, 128 filters, padding="same"
   net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), activation='relu',padding="same"))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

   # Add a convolutional layer, 3x3 window, 256 filters
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), activation='relu'))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

   # Flatten the output maps for fully connected layer
   net.add(tf.keras.layers.Flatten())

   # Add a fully connected layer of 128 neurons
   net.add(tf.keras.layers.Dense(units=128, activation='relu'))

   # Add a fully connected layer of 512 neurons
   net.add(tf.keras.layers.Dense(units=512, activation='relu'))

   # Add a fully connected layer with number of output neurons the same
   # as the number of classes
   net.add(tf.keras.layers.Dense(units = H*W*C, activation='linear'))
   net.add(tf.keras.layers.Reshape((H,W,C)))

   # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
   # training
   net.compile(optimizer='adam',
                 loss='mse')


   # Train the model for 50 epochs, using 33% of the data for validation measures,
   # shuffle the data into different batches after every epoch
   train_info = net.fit(x_train, x_train, validation_split=0.33,  epochs=50, shuffle=True)

   # Save the model to file
   print("Saving neural network to %s..." % net_save_name)
   net.save(net_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)

# *********************************************************
# * Training history *
# *********************************************************

# Plot training and validation accuracy over the course of training
if history != []:
   fh = plt.figure()
   ph = fh.add_subplot(111)
   ph.plot(history['loss'], label='mse')
   ph.plot(history['val_loss'], label = 'val_mse')
   ph.set_xlabel('Epoch')
   ph.set_ylabel('MSE loss')
   ph.set_ylim([0, 1])
   ph.legend(loc='lower right')

# *********************************************************
# * Train History *
# *********************************************************

if history != []:
   fh = plt.figure()
   ph = fh.add_subplot(111)
   ph.plot(history['loss'], label='mse')
   ph.plot(history['val_loss'], label = 'val_mse')
   ph.set_xlabel('Epoch')
   ph.set_ylabel('MSE loss')
   ph.legend(loc='lower right')


# Compute output for 16 test images
y_test = net.predict(x_test[:16])
y_test = y_test.astype(np.uint8)

# Show true labels and predictions for 16 test images
show_methods.show_data_images(images = y_test[:16], class_names=class_names, blocking = True)
