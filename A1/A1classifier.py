import tensorflow as tf
from load_smallnorb import load_smallnorb
import show_methods
import os
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time

def smallnorb_classifier(load_from_file = False, verbose = True, reg_wdecay_beta = 0, reg_dropout_rate = 0, reg_batch_norm = False, data_aug=False):
    t = time.time()
    (train_images, train_labels),(test_images, test_labels) = load_smallnorb()

    # Extract the class and elevation labels from the train and test labels
    train_classes = train_labels[:,2]
    test_classes = test_labels[:,2]

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    class_names = ['animal', 'human', 'airplane', 'truck', 'car']

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    # Specify the names of the save files
    save_name = os.path.join('saved', 'smallnorbclassify2norm2_rwd%.1e_rdp%.1f_rbn%d_daug%d' % (reg_wdecay_beta,reg_dropout_rate,int(reg_batch_norm),int(data_aug)))
    net_save_name = save_name + '_cnn_net.h5'
    checkpoint_save_name = save_name + '_cnn_net.chk'
    history_save_name = save_name + '_cnn_net.hist'
    # Show 16 train images with the corresponding labels
    if verbose:
        train_images_sample = train_images[:16,:,:,0]
        train_classes_sample = train_classes[:16]
        show_methods.show_data_images(images=train_images_sample, labels=train_classes_sample, class_names=class_names, blocking=False)

    n_classes = len(class_names)

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

        net.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same", input_shape=(96, 96, 2)))
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
        #net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        #net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        #net.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1,1), activation='relu', padding = "same"))
        #net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        net.add(tf.keras.layers.Flatten())
        net.add(tf.keras.layers.Dense(units=64, activation='relu'))
        net.add(tf.keras.layers.Dense(units=128, activation='relu'))
        #net.add(tf.keras.layers.Dense(units=256, activation='relu'))
        #net.add(tf.keras.layers.Dense(units=512, activation='relu'))
        net.add(tf.keras.layers.Dense(units=256, activation='relu'))
        net.add(tf.keras.layers.Dense(units=128, activation='relu'))
        net.add(tf.keras.layers.Dense(units=64, activation='relu'))
        if reg_dropout_rate > 0:
            # Dropout layer 1:
            net.add(tf.keras.layers.Dropout(reg_dropout_rate))
        net.add(tf.keras.layers.Dense(units=n_classes,activation='softmax'))
        
        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        net.compile(optimizer='sgd',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        
        # Training callback to call on every epoch -- evaluates
        # the model and saves its weights if it performs better
        # (in terms of accuracy) on validation data than any model
        # from previous epochs
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
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
            valid_classes = train_classes[I[:N_valid]]

            # Select the training input and the corresponding labels
            train_images = train_images[I[N_valid:]]
            train_classes = train_classes[I[N_valid:]]


            # Crate data generator that randomly manipulates images
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                zca_epsilon=1e-06,
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                horizontal_flip=True
            )

            # Configure the data generator for the images in the training sets
            datagen.fit(train_images)

            # Build the data generator
            train_data_aug = datagen.flow(train_images, train_classes)

            if verbose:
                print(train_images_sample.shape)
                print(train_classes_sample.shape)
                train_images_sample = train_images_sample.reshape(16,96,96,1)
                for x_batch, y_hat_batch in datagen.flow(train_images_sample, train_classes_sample, shuffle=False):
                    show_methods.show_data_images(images=x_batch.astype('uint8'), labels=y_hat_batch, class_names=class_names,
                                                    blocking=False)
                    break


            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model
            train_info = net.fit(train_data_aug,
                                validation_data=(valid_images, valid_classes),
                                epochs=100, shuffle=True,
                                callbacks=[model_checkpoint_callback])
        else:
            # Train the model for 50 epochs, using 33% of the data for validation measures,
            # shuffle the data into different batches after every epoch, include the checkpoint
            # callback that will save the best model
            train_info = net.fit(train_images, train_classes, validation_split=0.33,  epochs=20, shuffle=True,
                                callbacks=[model_checkpoint_callback])

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
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history['accuracy'], label='accuracy')
        ph.plot(history['val_accuracy'], label = 'val_accuracy')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('Accuracy')
        ph.set_ylim([0, 1])
        ph.legend(loc='lower right')

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************


    if verbose:
        loss_train, accuracy_train = net.evaluate(train_images, train_classes, verbose=0)
        loss_test, accuracy_test = net.evaluate(test_images, test_classes, verbose=0)

        print("Train accuracy (tf): %.2f" % accuracy_train)
        print("Test accuracy  (tf): %.2f" % accuracy_test)

        # Compute output for 16 test images
        y_test16 = net.predict(test_images[:16])
        y_test16 = np.argmax(y_test16, axis=1)
        y_test = net.predict(test_images)
        y_test = np.argmax(y_test, axis=1)
        confusion_matrix = tf.math.confusion_matrix(test_classes, y_test)
        print(confusion_matrix)
        print("trained in ", (time.time()-t) /60, " minutes")

        # Show true labels and predictions for 16 test images
        show_methods.show_data_images(images=test_images[:16,:,:,0],
                                        labels=test_classes[:16],predictions=y_test,
                                        class_names=class_names,blocking=True)



    return net

    
if __name__ == "__main__":
   smallnorb_classifier(load_from_file=True, verbose=True,
             reg_wdecay_beta=0.1, reg_dropout_rate=0.5, reg_batch_norm=True, data_aug=False)