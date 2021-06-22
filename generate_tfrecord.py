import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', metavar='', default='data')
parser.add_argument('-r', '--resize', metavar='', type=int, default=256)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(filename, label):
    image_string = open(filename, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_string).shape
    
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image': _bytes_feature(image_string),          
        'label': _int64_feature(label),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example

def readDatasetDF(datasetType, rootDir='data/'):
    """
    dataset Type: "train", "val", "test" 
    Return:
        dataframe with image name/label
    """
    df = pd.read_csv('{}/{}-calibrated-shuffled.txt'.format(rootDir, datasetType), header=None, delimiter = " ")
    df[0] = rootDir+"/"+df[0]
    return df

def main():
    df = readDatasetDF('train')
    writer = tf.io.TFRecordWriter("data/out/images.tfrecords")
    
    for i, img_info in df.iterrows():
        filename = img_info[0]
        label = img_info[1]
        tf_example = create_tf_example(filename, label)
        writer.write(tf_example.SerializeToString())

if __name__=='__main__':
    args = parser.parse_args()
    RESIZE = args.resize
    main()
    print('done')