import tensorflow as tf

def splitDataset(full_dataset, DATASET_SIZE):
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.1 * DATASET_SIZE)
    test_size = int(0.1 * DATASET_SIZE)

    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)
    
    return train_dataset, val_dataset, test_dataset 

def _extract_fn(tfrecord):
    # Extract features using the keys set during creation
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),        
        'label': tf.io.FixedLenFeature([], tf.int64)}

    sample = tf.io.parse_single_example(tfrecord, feature)

    image = tf.io.decode_raw(sample['image'], tf.uint8) 
    label = tf.cast(sample['label'], tf.int64)
    
    return [image, label] 

def batch_dataset(dataset, batch_size):
    dataset = dataset.map(_extract_fn)
    # dataset = dataset.prefetch()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset