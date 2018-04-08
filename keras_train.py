import os, time
# import h5py
import numpy as np
# from kerasimage import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential
# from keras.layers import Convolution2sing.D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from keras import backend as K

K.set_image_dim_ordering('th')          # 指定维度排序

# path to the model weights files.
weights_path = '/home/xjh/Downloads/FASNet/weights/REPLAY-ftweights18.h5'
# weights_path = '/home/xjh/Downloads/vgg16_weights.h5'     
# weights_path = None       # 若是没有预训练权重则设置为None
top_model_weights_path = './weight_final.h5'
img_width, img_height = (128, 128)
nb_epoch = 10      # 训练批次
# dimensions of images. (less than 224x 224)

# 要冻结的层数
nFreeze = 10

train_data_dir = '/home/xjh/Desktop/face/face/train/'
validation_data_dir ='/home/xjh/Desktop/face/face/val/'
# nb_train_samples = ()
# nb_validation_samples = ()


def get_tr_vgg_model(weights_path, img_width, img_height):

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    # assert os.path.exists(weights_path)

    'Model weights not found (see "weights_path" variable in script).'
    # f = h5py.File(weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         # we don't look at the last (fully-connected) layers in the savefile
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    print('Model loaded.')

    return model

def add_top_layers(model):

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # add the model on top of the convolutional base
    model.add(top_model)

    return model

def run_train(model):

    start_time = time.time()

    # freeze layers 从第十层开始训练
    for layer in model.layers[:nFreeze]:
        layer.trainable = False

    # compile model
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6),
              metrics=['accuracy'])

    print('Model Compiled.')

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=50,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=50,
            class_mode='binary')

    print('\nFine-tuning top layers...\n')

    earlyStopping = callbacks.EarlyStopping(monitor='val_acc',
                                           patience=10,
                                           verbose=0, mode='auto')
    #
    # #fit model
    '''
    fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
    validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
    use_multiprocessing=False, shuffle=True, initial_epoch=0)
    '''

    model.fit_generator(
           train_generator,
           callbacks=[earlyStopping],
            steps_per_epoch=len(train_generator),        # 200
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
        )
    #
    model.save_weights(top_model_weights_path)

    print('\nDone fine-tuning, have a nice day!')
    print("\nExecution time %s seconds" % (time.time() - start_time))


if __name__ == "__main__":

    vgg16_tr_model = get_tr_vgg_model(weights_path, img_width, img_height)
    vgg16_tr_model = add_top_layers(vgg16_tr_model)

    # fine-tuning the model
    run_train(vgg16_tr_model)

