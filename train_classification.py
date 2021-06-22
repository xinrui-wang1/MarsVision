import tensorflow as tf
import cv2
import glob
import numpy as np
from dataset_helper import *
from models import *


def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = loss_fn(labels, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(labels, preds)

def test_step(images, labels, lossFN, accFN):
    """
    lossFN: val_loss vs test_loss
    """
    preds = model(images, training=False)
    loss = loss_fn(labels, preds)
    lossFN(loss)
    accFN(labels, preds)

def main():
    # Split dataset
    tfrds = tf.data.TFRecordDataset(TFRECORD_PATH)
    DATASET_SIZE = sum(1 for record in tfrds)
    # train_tfr, val_tfr, test_tfr = splitDataset(tfrds, DATASET_SIZE)
    train_ds = batch_dataset(train_tfr, BATCH_SIZE)
    # val_ds = batch_dataset(val_tfr, BATCH_SIZE)
    # test_ds = batch_dataset(test_tfr, BATCH_SIZE)

    # Tensorboard
    train_log_dir = 'logs/{}/train'.format(EXP_NAME)
    val_log_dir = 'logs/{}/val'.format(EXP_NAME)
    test_log_dir = 'logs/{}/test'.format(EXP_NAME)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    best_train_loss = 1e5 
    best_test_loss = 1e5
    save_path_train = 'saved_models/{}_train/cp.ckpt'.format(EXP_NAME)
    save_path_test = 'saved_models/{}_test/cp.ckpt'.format(EXP_NAME)

    earlyStopTH = 0
    tempTH = 0
    prevValLoss = 0

    for epoch in range(EPOCHS):
        print("Epoch:", epoch+CKPT_EPOCH)
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        # train
        train_ds = train_ds.shuffle(buffer_size=1000)
        for cur_batch in train_ds:
            images = cur_batch[0] / 255
            images = tf.reshape(images, [-1, RESIZE, RESIZE, 1])
            images = tf.dtypes.cast(images, tf.float32)
            labels = cur_batch[1]
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch+CKPT_EPOCH)
            tf.summary.scalar('acc', train_acc.result(), step=epoch+CKPT_EPOCH)
        print('  Train loss:', train_loss.result().numpy())
        print('  Train acc:', train_acc.result().numpy())

        # val
        for cur_batch in val_ds:
            images = cur_batch[0] / 255
            images = tf.reshape(images, [-1, RESIZE, RESIZE, 1])
            images = tf.dtypes.cast(images, tf.float32)
            labels = cur_batch[1]
            test_step(images, labels, val_loss, val_acc)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch+CKPT_EPOCH)
            tf.summary.scalar('acc', val_acc.result(), step=epoch+CKPT_EPOCH)
        print('  Val loss:', val_loss.result().numpy())
        print('  Val acc:', val_acc.result().numpy())

        # test
        for cur_batch in test_ds:
            images = cur_batch[0] / 255
            images = tf.reshape(images, [-1, RESIZE, RESIZE, 1])
            images = tf.dtypes.cast(images, tf.float32)
            labels = cur_batch[1]
            test_step(images, labels, test_loss, test_acc)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch+CKPT_EPOCH)
            tf.summary.scalar('test', test_acc.result(), step=epoch+CKPT_EPOCH)
        print('  Test loss:', test_loss.result().numpy())
        print('  Test acc:', test_acc.result().numpy())


        # save best model
        # if train_loss.result().numpy() < best_train_loss:
        #     best_train_loss = train_loss.result().numpy()
        #     model.save_weights(save_path_train)

        if test_loss.result().numpy() < best_test_loss:
            best_val_loss = test_loss.result().numpy()
            # model.save_weights(save_path_test)
            tf.saved_model.save(model, save_path_test)

        # early stopping
        if val_loss.result().numpy() > prevValLoss:
            earlyStopTH += 1
            if earlyStopTH > EARLY_STOP:
                print("Early stopping at epoch", str(epoch+CKPT_EPOCH))
                break
        else:
            earlyStopTH = 0
        prevValLoss = val_loss.result().numpy()

if __name__ == "__main__":
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    TFRECORD_PATH = 'StraightData/train.record'
    BATCH_SIZE = 64
    EPOCHS = 1000
    RESIZE = 64
    EARLY_STOP = 5
    CKPT_PATH = 'saved_models/lr3e-4_test/cp.ckpt'
    EXP_NAME = 'lr3e-5'
    CKPT_EPOCH = 53

    if CKPT_PATH is None:
        model = StraightEvalModel()
    else:
        model = tf.saved_model.load(CKPT_PATH)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # default 0.001

    # Initialize metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.BinaryAccuracy(name='train_acc')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc = tf.keras.metrics.BinaryAccuracy(name='val_acc')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.BinaryAccuracy(name='test_acc')
    
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    main()