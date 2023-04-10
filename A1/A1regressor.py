import tensorflow as tf
from load_smallnorb import load_smallnorb
import show_methods
import os
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time

# Define the RMSE loss function
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def smallnorb_classifier(load_from_file = False, verbose = True, reg_wdecay_beta = 0, reg_dropout_rate = 0, reg_batch_norm = False, data_aug=False):
    t = time.time()
    (train_images, train_labels),(test_images, test_labels) = load_smallnorb()
    print(type(train_labels[:,3][0]))
    train_elevations = train_labels[:,3].astype(np.float32)
    test_elevations = test_labels[:,3].astype(np.float32)
    # Extract the class and elevation labels from the train and test labels

    train_elevations = train_elevations * 5 + 30
    test_elevations = test_elevations * 5 + 30
    print(type(train_elevations[0]))

    #normalize images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #store only first channel
    train_images = train_images[:,:,:,0]
    test_images = test_images[:,:,:,0]

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    # Specify the names of the save files
    save_name = os.path.join('saved', 'smallnorbelevation_rwd%.1e_rdp%.1f_rbn%d_daug%d' % (reg_wdecay_beta,reg_dropout_rate,int(reg_batch_norm),int(data_aug)))
    net_save_name = save_name + '_cnn_net.h5'
    checkpoint_save_name = save_name + '_cnn_net.chk'
    history_save_name = save_name + '_cnn_net.hist'
    # Show 16 train images with the corresponding labels
    if verbose:
        train_images_sample = train_images[:16,:,:]
        train_elevation_sample = train_elevations[:16]
        show_methods.show_data_images(images=train_images_sample, labels=train_elevation_sample, blocking=False)

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
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
        net.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same", input_shape=(96, 96, 1)))
        net.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())

        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        net.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())

        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        net.add(tf.keras.layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        if reg_batch_norm:
            # Batch norm layer 1
            net.add(tf.keras.layers.BatchNormalization())
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Flatten())
        if reg_wdecay_beta > 0:
            reg_wdecay = tf.keras.regularizers.l2(reg_wdecay_beta)
        else:
            reg_wdecay = None
        net.add(tf.keras.layers.Dense(units=64, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=2048, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=2048, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=2048, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=512, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=256, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=128, activation='sigmoid', kernel_regularizer=reg_wdecay))
        net.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=reg_wdecay))

        if reg_dropout_rate > 0:
            # Dropout layer 1:
            net.add(tf.keras.layers.Dropout(reg_dropout_rate))
        net.add(tf.keras.layers.Dense(units=1, activation='linear'))
        
        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        
        net.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mean_squared_error'])
        
        # Training callback to call on every epoch -- evaluates
        # the model and saves its weights if it performs better
        # (in terms of accuracy) on validation data than any model
        # from previous epochs
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor='val_mean_squared_error',
            mode='min',
            save_best_only=True)
        
        if data_aug:
            # Read the number of training points - len on numpy array returns the number of rows...
            N = len(train_images)
            # Specify the number of points to use for validation
            N_valid = int(N*0.33)

            # Generate a list of randomly ordered indexes from 0 to N-1
            I = np.random.permutation(N)

            # Select the validation inputs and the corresponding labels
            valid_images = train_images[I[:N_valid]]
            valid_elevations = train_elevations[I[:N_valid]]

            # Select the training input and the corresponding labels
            train_images = train_images[I[N_valid:]]
            train_elevations = train_elevations[I[N_valid:]]


            # Crate data generator that randomly manipulates images
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                zca_epsilon=1e-06,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                horizontal_flip=True
            )

            # Configure the data generator for the images in the training sets
            train_images = train_images.reshape(train_images.shape[0],96,96,1)
            datagen.fit(train_images)

            # Build the data generator
            train_data_aug = datagen.flow(train_images, train_elevations)

            if verbose:
                print(train_images_sample.shape)
                print(train_elevation_sample.shape)
                train_images_sample = train_images_sample.reshape(16,96,96,1)
                for x_batch, y_hat_batch in datagen.flow(train_images_sample, train_elevation_sample, shuffle=False):
                    #show_methods.show_data_images(images=x_batch.astype('uint8'), labels=y_hat_batch, class_names=class_names,blocking=False)
                    break


            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model
            train_info = net.fit(train_data_aug,
                                validation_data=(valid_images, valid_elevations),
                                epochs=5, shuffle=True,
                                callbacks=[model_checkpoint_callback], batch_size=32)
        else:
            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model
            train_info = net.fit(train_images, train_elevations, validation_split=0.33,  epochs=1000, shuffle=False,
                                callbacks=[model_checkpoint_callback], batch_size=32)
        

        #train_info = net.fit(train_images, train_elevations, validation_split=0.33,  epochs=20, shuffle=True, callbacks=[model_checkpoint_callback], batch_size=32)

        # Load the weights of the best model
        print("Loading best save weight from %s..." % checkpoint_save_name)
        net.load_weights(checkpoint_save_name)

        # Save the entire model to file
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
    if verbose and history != []:
        fh = plt.figure(num = "training history")
        history = train_info.history
        ph = fh.add_subplot(111)
        ph.plot(history['loss'], label='MSE')
        ph.plot(history['val_loss'], label = 'val_MSE')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('MSE')
        ph.set_ylim([0, 500])
        ph.legend(loc='lower right')
        #plt.show()

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************


    if verbose:
        loss_train, mae_train = net.evaluate(train_images, train_elevations, verbose=0)
        loss_test, mae_test = net.evaluate(test_images, test_elevations, verbose=0)
        print(test_images.shape[0])
        pred = net.predict(test_images)
        print(pred.size)
        print(test_elevations[:16])
        print(pred[:16])

        print("Train MAE (tf): %.2f" % mae_train)
        print("Test MAE  (tf): %.2f" % mae_test)

        # Compute output for 16 test images
        y_test16 = net.predict(test_images[:16])
        y_test16 = np.argmax(y_test16, axis=1)
        y_test = net.predict(test_images)
        y_test = np.argmax(y_test, axis=1)
        print("trained in ", (time.time()-t) /60, " minutes")

        # Show true labels and predictions for 16 test images
        #show_methods.show_data_images(images=test_images[:16,:,:], labels=test_elevations[:16],predictions=y_test, blocking=True)

    return net

    
if __name__ == "__main__":
   smallnorb_classifier(load_from_file=True, verbose=True, reg_wdecay_beta=0.0, reg_dropout_rate=0.0, reg_batch_norm=False, data_aug=False)