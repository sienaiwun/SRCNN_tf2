import time
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from nnconfig import srcnn_flags
from tensorflow.keras import datasets, layers, models
from utils import (
        prepare_data,
        preprocess,
        show_gray_image,
        show_image,
        save_data,
        read_data,
        merge
)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model):
        self.counter = 0
        self.ep = 0
        self.srcnn_model = model
        self.start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.ep += 1

    def on_train_batch_end(self, batch, logs=None):
        self.counter += 1

        if self.counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%s]" \
                  % ((self.ep + 1), self.counter, time.time() - self.start_time, logs))
        if self.counter % 500 == 0:
            file_name = self.srcnn_model.flags.checkpoint_dir
            print("save path" + file_name)
            self.srcnn_model.save(file_name)

class SRCNN(object):
    def __init__(self,
                 flags,
                 ):
        self.flags = flags
        self.build_model()

    def build_model(self):
        if self.flags.is_train:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(64, (9, 9), activation='relu', input_shape=(self.flags.image_size, self.flags.image_size, self.flags.c_dim)))
            self.model.add(layers.Conv2D(32, (1, 1), activation='relu'))
            self.model.add(layers.Conv2D(1, (5, 5), activation='relu'))
            optimizer = keras.optimizers.Adam(lr=self.flags.learning_rate)
            self.model.compile(optimizer=optimizer,
                               loss="mse",
                               metrics=['mse'])
        else:
            self.model = self.load(self.flags.checkpoint_dir)
        self.model.summary()


    def train(self):
        if self.flags.is_train:
            self.input_setup()
        else:
            nx, ny = self.input_setup()
        train_data, train_label = read_data(self.data_dir())
        if self.flags.is_train:
            self.model.fit(x= train_data,y= train_label,epochs= self.flags.epoch, batch_size= self.flags.batch_size,callbacks=[CustomCallback(self)])
        else:
            results = self.model.predict(x = train_data)
            result = merge(results, [nx, ny])
            #show_image(result)

    def data_dir(self):
        if self.flags.is_train:
            return os.path.join('./{}'.format(self.flags.checkpoint_dir), "train.h5")
        else:
            return os.path.join('./{}'.format(self.flags.checkpoint_dir), "test.h5")


    def input_setup(self):
        if self.flags.is_train:
            imgs_path_sequence = prepare_data(self.flags, data_path="Train")
        else:
            imgs_path_sequence = prepare_data(self.flags, data_path="Test")
        inputs_sequence = []
        lables_sequence = []
        padding = abs(self.flags.image_size - self.flags.label_size) / 2

        if self.flags.is_train:
            for i in range(len(imgs_path_sequence)):
                input_, label_ = preprocess(imgs_path_sequence[i], self.flags.scale, self.flags.is_grayscale)
                #show_image(input_)
                #show_image(label_)
                if len(input_.shape) == 3:
                    h, w, _ = input_.shape
                else:
                    h, w = input_.shape

                for x in range(0, h - self.flags.image_size + 1, self.flags.stride):
                    for y in range(0, w - self.flags.image_size + 1, self.flags.stride):
                        sub_input = input_[x:x + self.flags.image_size, y:y + self.flags.image_size]  # [33 x 33]
                        sub_label = label_[x+int(padding):x+int(padding)+self.flags.label_size, y+int(padding):y+int(padding)+self.flags.label_size] # [21 x 21]
                        sub_input = sub_input.reshape([self.flags.image_size, self.flags.image_size, self.flags.c_dim])
                        sub_label = sub_label.reshape([self.flags.label_size, self.flags.label_size, self.flags.c_dim])

                        inputs_sequence.append(sub_input)
                        lables_sequence.append(sub_label)
                        
                        #show_image(sub_input[int(padding):int(padding)+self.flags.label_size,int(padding):int(padding)+self.flags.label_size])
                        #show_image(sub_label)
        else:
            input_, label_ = preprocess(imgs_path_sequence[1], self.flags.scale, self.flags.is_grayscale)
            #show_image(label_)
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            nx = ny = 0
            for x in range(0, h - self.flags.image_size + 1, self.flags.stride):
                nx += 1;
                ny = 0
                for y in range(0, w - self.flags.image_size + 1, self.flags.stride):
                    ny += 1
                    sub_input = input_[x:x + self.flags.image_size, y:y + self.flags.image_size]  # [33 x 33]
                    sub_label = label_[x + int(padding):x + int(padding) + self.flags.label_size,
                                y + int(padding):y + int(padding) + self.flags.label_size]  # [21 x 21]
                    sub_input = sub_input.reshape([self.flags.image_size, self.flags.image_size, self.flags.c_dim])
                    sub_label = sub_label.reshape([self.flags.label_size, self.flags.label_size, self.flags.c_dim])
                    #show_image(sub_label)
                    inputs_sequence.append(sub_input)
                    lables_sequence.append(sub_label)

        arrdata = np.asarray(inputs_sequence)
        arrlabel = np.asarray(lables_sequence)
        save_path = self.data_dir()
        save_data(arrdata, arrlabel, save_path)
        if not self.flags.is_train:
            return nx, ny

    def model_dir(self):
        return "%s_%s_%d" % ("srcnn", self.flags.label_size,self.flags.c_dim)

    def save(self, checkpoint_dir):
        model_dir = self.model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model.save(checkpoint_dir)

    def load(self, checkpoint_dir):
        model_dir = self.model_dir()
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        return tf.keras.models.load_model(checkpoint_dir)