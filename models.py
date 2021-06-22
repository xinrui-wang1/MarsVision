import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout

RESIZE = 64

class CenterCorrectionModel(tf.keras.Model):
    def __init__(self, training=True):
        super(CenterCorrectionModel, self).__init__()
        self.training=training
        
        self.conv1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
        self.b1 = BatchNormalization()
        self.conv2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
        self.b2 = BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.d1 = Dense(128, activation='relu')
        
        self.r_layer = Dense(1, activation='linear')
        self.theta_layer = Dense(1, activation='linear')

    def call(self, x):
        x = self.b1(self.conv1(x))
        x = self.b2(self.conv2(x))
        x = self.global_pool(x)

        x = self.d1(x)
        
        r_pred = self.r_layer(x)
        theta_pred = self.theta_layer(x)
        return r_pred, theta_pred

class StraightEvalModel(tf.keras.Model):
    def __init__(self, training=True, input_shape=[RESIZE, RESIZE, 1]):
        super(StraightEvalModel, self).__init__()
        self.training=training
        
        self.conv1 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=input_shape)
        self.b1 = BatchNormalization()
        self.conv2 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
        self.b2 = BatchNormalization()
        # self.conv3 = Conv2D(filters=128, kernel_size=(3,3), activation='relu')
        # self.b3 = BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.d1 = Dense(128, activation='relu')
        
        self.d2 = Dense(1, activation='linear') # update 

    def call(self, x):
        x = self.b1(self.conv1(x))
        x = self.b2(self.conv2(x))
        # x = self.b3(self.conv3(x))
        x = self.global_pool(x)

        x = self.d1(x)
        x = self.d2(x)
        
        return x