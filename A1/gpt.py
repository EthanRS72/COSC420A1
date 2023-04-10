import tensorflow as tf

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Preprocess data
x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)
y_train = (y_train - y_train.mean()) / y_train.std()
y_test = (y_test - y_train.mean()) / y_train.std()

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Train model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate model on test set
loss, mae = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, test MAE: {mae}')



