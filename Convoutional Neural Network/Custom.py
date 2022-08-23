from createArray import importArrays
from tensorflow import keras
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.metrics import SpecificityAtSensitivity
from createTFRecord import createTF, createTestTF
from predictClass import predict
from displayResults import lossCurve
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
     tf.config.experimental.set_memory_growth(gpu, True)

noMS_traindir = "Gen 2/MRI Data/Training/noMS"
MS_traindir = "Gen 2/MRI Data/Training/MS"
noMS_testdir = "Gen 2/MRI Data/Test/noMS"
MS_testdir = "Gen 2/MRI Data/Test/MS"
height, width, depth = 181, 217, 181
batch_size = 2 # number of samples that will be propagated through the network. Less = less memory but more inaccurate
prefetch_size = 1 # How many data sets to prefetch for gpu, increase until no speed increases
ratio = 0.7 # ratio of training to validation
printPredict = True # Print each prediction
modelnum = 1 # Select which model to use
epochs = 50 # Number of trainings

# Optimizer
OptimizerType = "Adam" # Adam, SGD, RMSprop
momentum = 0.0 # SGD, RMSprop

# Modify Learning Rate: https://keras.io/api/optimizers/learning_rate_schedules/, for info on what these mean.
Learnertype = "ExponentialDecay" # ExponentialDecay, PolynomialDecay, InverseTimeDecay, Constant
initial_learning_rate = 0.0002 # Needed for ExponentialDecay, PolynomialDecay, InverseTimeDecay, Constant
end_learning_rate = 0.0001 # Needed for PolynomialDecay
decay_steps = 100000 # Needed for ExponentialDecay, PolynomialDecay, InverseTimeDecay
power = 0.5 # Needed for PolynomialDecay
decay_rate = 0.90 # Needed for ExponentialDecay, InverseTimeDecay
staircase = True # Needed for ExponentialDecay, InverseTimeDecay

# Callbacks
checkpoint = True # Use checkpoint callback to save best
earlystopping = True # Stop model early if no growth
monitor = "val_loss" # What quantity to monitor for early stopping
patience = 30 # Epochs with no improvement, training will be stopped.
factorReduce = 0.2 # Factor learning rate should be reduced after stagnation
patienceLR = 5 # Epochs after learning rate should be reduced
min_lr = 0.00001 # Minimum learning rate

from Models import get_model_1, get_model_2, get_model_3, get_model_4, get_model_5, get_model_6, get_model_7

inputstring = input("Enter C for create TF Record. Enter T for Train. Enter P for predict. : ")
inputstring = inputstring.lower()
if inputstring.find('c') != -1:
    createTF(noMS_traindir, MS_traindir, height, width, depth)
    print("Done creating Train TFRecord")
    createTestTF(noMS_testdir, MS_testdir, height, width, depth)
    print("Done creating Test TFRecord")
if inputstring.find('t') != -1:
    train_dataset, validation_dataset = importArrays(batch_size, prefetch_size, ratio)
    model = None
    if modelnum == 1:
        model = get_model_1(height, width, depth)
    if modelnum == 2:
        model = get_model_2(height, width, depth)
    if modelnum == 3:
        model = get_model_3(height, width, depth)
    if modelnum == 4:
        model = get_model_4(height, width, depth)
    if modelnum == 5:
        model = get_model_5(height, width, depth)
    if modelnum == 6:
        model = get_model_6(height, width, depth)
    if modelnum == 7:
        model = get_model_7(height, width, depth)
    model.summary()

    # Learning rate overtime
    
    if Learnertype == "ExponentialDecay":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
    elif Learnertype == "PolynomialDecay":
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, decay_steps=decay_steps, end_learning_rate=end_learning_rate, power=power)
    elif Learnertype == "InverseTimeDecay":
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)
    elif Learnertype == "Constant":
        lr_schedule = initial_learning_rate

    METRICS = [
    #   TruePositives(name='TruePos'),
    #   FalsePositives(name='FalsePos'),
    #   TrueNegatives(name='TrueNeg'),
    #   FalseNegatives(name='FalseNeg'), 
      BinaryAccuracy(name='Accuracy'),
      Precision(name='Precision'),
      Recall(name='Recall'),
      AUC(name='AUC'),
      SpecificityAtSensitivity(sensitivity=0.8, name='Sensitivity'),
    ]

    # Creating Model
    if OptimizerType == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule) # Type of optimizer used, there are many: https://keras.io/api/optimizers/
    if OptimizerType == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum) # Type of optimizer used, there are many: https://keras.io/api/optimizers/
    if OptimizerType == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule, momentum=momentum) # Type of optimizer used, there are many: https://keras.io/api/optimizers/
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=METRICS,
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "Gen 2/Code/CNN/Custom/bestClassification.h5", save_best_only=True, monitor=monitor
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=1, restore_best_weights=True) # Stop early if no change, patience is how long it should wait
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factorReduce, patience=patienceLR, min_lr=min_lr)


    callbacks = []
    if checkpoint & earlystopping:
        callbacks = [checkpoint_cb, early_stopping_cb]
    elif checkpoint:
        callbacks = [checkpoint_cb]
    elif earlystopping:
        callbacks = [early_stopping_cb]

    # Train the model, doing validation at the end of each epoch
    history = model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    model.save("Gen 2/Code/CNN/Custom/currClassification.h5")
    lossCurve(history.history)

    print("Done Computing")
if inputstring.find('p') != -1:
    predict(printPredict, batch_size, prefetch_size)