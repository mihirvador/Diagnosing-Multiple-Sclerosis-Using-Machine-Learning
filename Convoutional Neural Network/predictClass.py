import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
     tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

def decode(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features={'train/image': tf.io.FixedLenFeature([181, 217, 181, 1], tf.float32),
                  'train/label': tf.io.FixedLenFeature([], tf.int64)})

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    img = features['train/image']
    label = tf.cast(features['train/label'], tf.int32)
    return img, label

def predict(printPredict, batch_size, prefetch_size):
    model = tf.keras.models.load_model("Gen 2/Code/CNN/Custom/bestClassification.h5")
    print("Starting Prediction")

    length = sum(1 for _ in tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/testset.tfrecords"))
    print("Length is: " + str(length))
    # Define data loaders.
    full_dataset = tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/testset.tfrecords").map(decode)
    test_dataset = (
            full_dataset.shuffle(batch_size * 10)
            .batch(batch_size)
            .prefetch(prefetch_size)
        )

    prediction = model.predict(x=test_dataset, verbose=1)
    results = []
    i = 0
    for next_element in test_dataset:
        text = next_element[1].numpy()
        label = text[0]
        label = "Actual Value: " + str(label)
        results.append([prediction[i][0], label])
        i = i + 1
    results = np.array(results, dtype='U')
    if printPredict:
        print(results)
    np.savetxt("Gen 2/Code/CNN/Custom/prediction.txt", results, fmt="%10s %10s")
    model.evaluate(x=test_dataset, verbose=1)