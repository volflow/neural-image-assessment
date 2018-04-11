import numpy as np
import glob
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_resnet_v2 import preprocess_input
from utils.nasnet import NASNetMobile
from utils.nasnet import preprocess_input as preprocess_input2
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# import IPython
# clear_output = IPython.core.display.clear_output

from utils.score_utils import mean_score, std_score

weigths_path = "./weights/mobilenet_weights.h5"

def evaluate(model,imgs):
    target_size = (224, 224)
    with tf.device('/CPU:0'):
        score_list = []
        for i,img_path in enumerate(imgs):
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input2(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)

            file_name = img_path.lower()
            score_list.append(scores)#(file_name, mean, std))

            print('\rEvaluating: {}/{}'.format(i,len(imgs)), end='')

            #print("Evaluating : ", img_path)
            #print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
            #print()

    return score_list

def mobilenet():
    with tf.device('/CPU:0'):
        base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights(weigths_path)
        return model

def nasnet():
    #needs 224x224 images
    with tf.device('/CPU:0'):
        base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/nasnet_weights.h5')
        return model

def inceptionnet():
    with tf.device('/CPU:0'):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/inception_resnet_weights.h5')
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')

    parser.add_argument('-img', type=str, default=[None], nargs='+',
                        help='Pass one or more image paths to evaluate them')

    parser.add_argument('-model', type=str, default='mobilenet',
                        help='Pass model that will be used for evaluation')

    parser.add_argument('-resize', type=str, default='false',
                        help='Resize images to 224x224 before scoring')

    parser.add_argument('-rank', type=str, default='true',
                        help='Whether to rank the images after they have been scored')

    parser.add_argument('-rename', type=str, default='true',
                        help='appends the rank in front of the filename')

    args = parser.parse_args()
    resize_image = args.resize.lower() in ("true", "yes", "t", "1")
    target_size = (224, 224) if resize_image else None
    rank_images = args.rank.lower() in ("true", "yes", "t", "1")
    rename_files = args.rename.lower() in ("true", "yes", "t", "1")

    # give priority to directory
    if args.dir is not None:
        print("Loading images from directory : ", args.dir)
        imgs = glob.glob(args.dir + '/[!.]*.png')
        imgs += glob.glob(args.dir + '/[!.]*..jpg')
        imgs += glob.glob(args.dir + '/[!.]*.*.jpeg')

        for img_path in imgs:
            print(img_path)

    elif args.img[0] is not None:
        print("Loading images from path(s) : ", args.img)
        imgs = args.img

    else:
        raise RuntimeError('Either -dir or -img arguments must be passed as argument')

    if args.model == 'mobilenet':
        model = mobilenet()
    elif args.model == 'inceptionnet':
        model = inceptionnet()
    elif model == 'nasnet':
        model = nasnet()
    else:
        raise RuntimeError('Invalid model passed, try mobilenet, inceptionnet or naset')

    score_list = evaluate(model,imgs)
    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

            if rename_files:
                old_file = os.path.join(args.dir, name)
                new_file = os.path.join(args.dir, "{}_{}".format(i+1,name))
                os.rename(old_file, new_file)
