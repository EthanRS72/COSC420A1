import tensorflow as tf
from load_smallnorb import load_smallnorb
import show_methods
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import time


def apply_diffusion(images, timesteps=100, noise_std=0.1):
    # Create a copy of the input array to modify
    result = np.copy(images)

    # Loop over the number of diffusion steps
    for i in range(timesteps):
        #x = i +1
        print("Timestamp:",i+1)
        # Compute the standard deviation for the Gaussian noise
        std = noise_std * np.sqrt(i + 1)

        # Generate random Gaussian noise with the same shape as the input
        noise = np.random.normal(loc=0.0, scale=std, size=result.shape)

        # Add the noise to the input array
        result += noise

    # Return the result
    return result



def diffusion_generation(load_from_file = False, verbose = True, reg_wdecay_beta = 0, reg_dropout_rate = 0, reg_batch_norm = False, data_aug=False):
    t = time.time()
    (train_images, train_labels),(test_images, test_labels) = load_smallnorb()
    train_images = train_images.astype(np.float64) / 255.0
    train_images1 = train_images[:,:,:,0]
    train_images2 = train_images[:,:,:,1]
    train_images = np.concatenate((train_images1,train_images2), axis=0)
    train_images = train_images[:16]
    print(train_images.shape)
    print("diffusing images...")
    diffused_images = apply_diffusion(train_images, timesteps=5, noise_std=0.1)
    for i in range(len(diffused_images)):
        image = diffused_images[i]
        max = np.max(image)
        min = np.min(image)
        diffused_images[i] = (image - min) / (max - min)
    train_images = train_images.reshape(-1,96,96,1)
    diffused_images = diffused_images.reshape(-1,96,96,1)
    print("diffusing images done in ", (time.time()-t) /60, " minutes")

    # Display the original and diffused images side by side
    fig, axs = plt.subplots(8, 2, figsize=(10, 20))
    axs[0, 0].set_title('Original Image')
    axs[0, 1].set_title('Diffused Image')

    for i in range(8):
        # Display original image
        axs[i, 0].imshow(train_images[i], cmap='gray')
        # Display diffused image
        axs[i, 1].imshow(diffused_images[i], cmap='gray')
    plt.show()

    print(diffused_images.shape)
    print(diffused_images[0].shape)

    

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    # Specify the names of the save files
    save_name = os.path.join('saved', 'smallnorbdiffusion_rwd%.1e_rdp%.1f_rbn%d_daug%d' % (reg_wdecay_beta,reg_dropout_rate,int(reg_batch_norm),int(data_aug)))
    net_save_name = save_name + '_cnn_net.h5'
    checkpoint_save_name = save_name + '_cnn_net.chk'
    history_save_name = save_name + '_cnn_net.hist'


    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        generation_model = tf.keras.models.load_model(net_save_name)

        # Load the training history - since it should have been created right after
        # saving the model
        print("finding history")
        if os.path.isfile(history_save_name):
            print("history found")
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            print("history not found")
            history = []
    else:
      # ************************************************
      # * Creating and training a neural network model *
      # ************************************************
        t1 = time.time()
        # Create feed-forward network
        inputs = tf.keras.layers.Input(shape=(96,96,1))
        downconv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(inputs)
        downconv12nd = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides = (1,1), activation='relu', padding='same')(downconv1)
        downconv1pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(downconv12nd)
        outputs1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation=None) (downconv1pool)
        outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(outputs1)
        

        generation_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(generation_model.input)
        
        
        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        generation_model.compile(optimizer='sgd',
                        loss='mean_squared_error',
                        metrics=['mean_squared_error'])
        
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
        
        train_info = generation_model.fit(train_images, train_images, validation_split=0.1,  epochs=2, shuffle=True, callbacks=[model_checkpoint_callback], batch_size=16)
        print("trained in ", (time.time()-t1) /60, " minutes")
    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************


        if verbose:
            print("------------PREDICTING----------------")
            loss_train, mse_train = generation_model.evaluate(train_images, train_images, verbose=0)
            #loss_test, mse_test = generation.evaluate(test_images, test_elevations, verbose=0)

            print("Train mse: %.2f" % mse_train)
            #print("Test mse: %.2f" % mse_test)

            # Load the weights of the best model
            print("Loading best save weight from %s..." % checkpoint_save_name)
            generation_model.load_weights(checkpoint_save_name)

            # Save the entire model to file
            print("Saving neural network to %s..." % net_save_name)
            generation_model.save(net_save_name)

            # Save training history to file
            history = train_info.history
            with gzip.open(history_save_name, 'w') as f:
                pickle.dump(history, f)

    if verbose:
        print("------------PREDICTING----------------")
        loss_train, mse_train = generation_model.evaluate(diffused_images, train_images, verbose=0)
        #loss_test, mse_test = generation.evaluate(test_images, test_elevations, verbose=0)

        print("Train mse: %.2f" % mse_train)
        #print("Test mse: %.2f" % mse_test)
        generative_images = generation_model.predict(diffused_images)
        print(generative_images.shape)

        fig, axs = plt.subplots(8, 2, figsize=(10, 20))
        axs[0, 0].set_title('Original Image')
        axs[0, 1].set_title('Diffused Image')

        for i in range(8):
            # Display original image
            axs[i, 0].imshow(train_images[i], cmap='gray')
            # Display diffused image
            axs[i, 1].imshow(diffused_images[i], cmap='gray')
        plt.show()

    # *********************************************************
    # * Training history *
    # *********************************************************

    # Plot training and validation accuracy over the course of training
    if verbose and history != []:
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history['loss'], label='loss')
        ph.plot(history['val_loss'], label = 'val_loss')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('Loss')
        ph.set_ylim([0, 1])
        ph.legend(loc='lower right')
        

    return generation_model

    
if __name__ == "__main__":
   diffusion_generation(load_from_file=True, verbose=True, reg_wdecay_beta=0.0, reg_dropout_rate=0.0, reg_batch_norm=False, data_aug=False)

