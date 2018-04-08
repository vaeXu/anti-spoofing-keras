from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential
# from keras.layers import Convolution2sing.D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import callbacks
from keras import backend as K
import os
import time
from keras.utils import multi_gpu_model
K.set_image_dim_ordering('th')  # 指定维度排序

def load_model(weightsPath, img_width, img_height):

    # VGG-16 model
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

    # Top-model for anti-spoofing
    top_model = Sequential()
    # top_model = multi_gpu_model()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model.add(top_model)

    if weightsPath:
        model.load_weights(weightsPath)
    else:
        print('Could not load model!')

    return model
def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    imgArray = imgArray / float(255)
    return imgArray
def test(path):
    # img_width, img_height = (120, 120)
    img_width, img_height = (128, 128)
    # load weights
    # top_model_weights_path = '/home/xjh/Downloads/FASNet/weights/REPLAY-ftweights18.h5'
    top_model_weights_path = './weight_final.h5'
    # predict Class
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model = load_model(top_model_weights_path, img_width, img_height)
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    img = read_preprocess_image(path, img_width, img_height)
    outLabel = int(model.predict_classes(img, verbose=0))
    # print(outLabel)
    return outLabel

    # for file in os.listdir(path):
    #     # t1 = time.time()
    #     path_jpg = os.path.join(path, file)
    #     img = read_preprocess_image(path_jpg, img_width, img_height)
    #     outLabel = int(model.predict_classes(img, verbose=0))
    #     print(outLabel)
        # read and Pre-processing image
        # img = read_preprocess_image(path_jpg, img_width, img_height)
        # outLabel = int(model.predict_classes(img, verbose=0))
        # if outLabel == 0:
        #     print('真实照片 == ', path_jpg)
        #     i += 1
        #     print(i)
        # if outLabel == 1:
        #
        #     print('攻击图片 == ', path_jpg)
        # print(time.time() - t1)
        # return outLabel

def just(path):
    out = test(path)
    if out == 0:
        print('真')
    if out == 1:
        print('假')
if __name__ == '__main__':
    path = '/home/xjh/Desktop/jhd_false_test/0_106.jpg'
    just(path)

