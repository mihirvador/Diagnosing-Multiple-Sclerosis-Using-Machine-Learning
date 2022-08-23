import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

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


def importArrays(batch_size, prefetch_size, ratio):
    length = sum(1 for _ in tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/dataset.tfrecords"))
    lenRatio = int(length*ratio)
    # Define data loaders.
    full_dataset = tf.data.TFRecordDataset("Gen 2/Code/CNN/Custom/dataset.tfrecords").map(decode)
    full_dataset = full_dataset.shuffle(length)
    print("Done Shuffling")
    train_loader = full_dataset.take(lenRatio)
    validation_loader = full_dataset.skip(lenRatio)
    print("Data loaded")

    # Augment on the fly during training.
    train_dataset = (
        train_loader.shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(prefetch_size)
    )

    validation_dataset = (
        validation_loader.shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(prefetch_size)
    )
    print("Done dataset Preprocessing")

    return train_dataset, validation_dataset