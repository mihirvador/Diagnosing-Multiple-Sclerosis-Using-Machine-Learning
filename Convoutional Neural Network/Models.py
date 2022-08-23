from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotNormal
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
     tf.config.experimental.set_memory_growth(gpu, True)


# https://keras.io/examples/vision/mnist_convnet/
# https://cs230.stanford.edu/projects_spring_2018/reports/8291133.pdf
# https://towardsdatascience.com/building-a-brain-tumor-classification-app-e9a0eb9f068

def get_model_1(height, width, depth):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((height, width, depth, 1))

    x = layers.Conv3D(filters=8, kernel_size=3)(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=8, kernel_size=3)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=16, kernel_size=3)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=64)(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_2(height, width, depth):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((height, width, depth, 1))
    x = layers.MaxPool3D(pool_size=3)(inputs)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_3(height, width, depth):
    # https://ecode.dev/cnn-for-medical-imaging-using-tensorflow-2/

    inputs = keras.Input((height, width, depth, 1))

    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001), activation='relu')(inputs)
    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001), activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(units=1, kernel_initializer=GlorotNormal(), activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_4(height, width, depth):
    inputs = keras.Input((height, width, depth, 1))

    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_5(height, width, depth):
    inputs = keras.Input((height, width, depth, 1))

    x = layers.MaxPool3D(pool_size=3)(inputs)

    x = layers.Conv3D(filters=32, kernel_size=3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    

    x = layers.Conv3D(filters=64, kernel_size=3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
   
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_6(height, width, depth):
    inputs = keras.Input((height, width, depth, 1))
    x = layers.Conv3D(filters=4, kernel_size=3)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def get_model_7(height, width, depth):

    inputs = keras.Input((height, width, depth, 1))
    x = layers.Flatten()(inputs)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
