# Import the required modules
import tensorflow as tf
from tensorflow.keras import models, layers

# Initialize the Sequential model
model = models.Sequential([
   # Add a Convolutional layer with 12 filters, a kernel size of 3, stride of 1, padding set to 'same', and input shape of (224, 224, 3)
   layers.Conv2D(12, kernel_size=3, strides=1, padding='same', input_shape=(224, 224, 3)),
   # Add a Batch Normalization layer
   layers.BatchNormalization(),
   # Add a ReLU activation function
   layers.Activation('relu'),
   # Add a Max Pooling layer with pool size of 2
   layers.MaxPooling2D(pool_size=2),

   # Add another Convolutional layer with 20 filters, a kernel size of 3, stride of 1, and padding set to 'same'
   layers.Conv2D(20, kernel_size=3, strides=1, padding='same'),
   # Add a ReLU activation function
   layers.Activation('relu'),
   # Add a Max Pooling layer with pool size of 2
   layers.MaxPooling2D(pool_size=2),

   # Add another Convolutional layer with 32 filters, a kernel size of 3, stride of 1, and padding set to 'same'
   layers.Conv2D(32, kernel_size=3, strides=1, padding='same'),
   # Add a Batch Normalization layer
   layers.BatchNormalization(),
   # Add a ReLU activation function
   layers.Activation('relu'),
   # Add a Max Pooling layer with pool size of 2
   layers.MaxPooling2D(pool_size=2),

   # Add another Convolutional layer with 64 filters, a kernel size of 3, stride of 1, and padding set to 'same'
   layers.Conv2D(64, kernel_size = 3, strides = 1, padding = "same"),
   # Add a Batch Normalization layer
   layers.BatchNormalization(),
   # Add a ReLU activation function
   layers.Activation('relu'),
   # Add a Max Pooling layer with pool size of 2
   layers.MaxPooling2D(pool_size = 2),

   # Add another Convolutional layer with 128 filters, a kernel size of 3, stride of 1, and padding set to 'same'
   layers.Conv2D(128, kernel_size = 3, strides = 1, padding = "same"),
   # Add a Batch Normalization layer
   layers.BatchNormalization(),
   # Add a ReLU activation function
   layers.Activation('relu'),
   # Add a Max Pooling layer with pool size of 2
   layers.MaxPooling2D(pool_size = 2),

   # Flatten the output of the previous layer
   layers.Flatten(),
   # Add a Dense layer with 360 neurons and ReLU activation function
   layers.Dense(360, activation='relu'),
   # Add another Dense layer with 120 neurons and ReLU activation function
   layers.Dense(120, activation='relu'),
   # Add the final Dense layer with 'num_classes' neurons and softmax activation function
   layers.Dense(num_classes, activation = "softmax")
])
