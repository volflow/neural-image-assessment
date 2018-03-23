import numpy as np
from path import Path
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# from utils.score_utils import mean_score, std_score

weigths_path = "/Users/valentinwolf/Documents/programming/Python/NIMA/neural-image-assessment/weights/mobilenet_weights.h5"

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

def evaluate(model,imgs):
    target_size = (224, 224)
    with tf.device('/CPU:0'):
        score_list = []

        for img_path in imgs:
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)

            file_name = Path(img_path).name.lower()
            score_list.append((file_name, mean, std))

            print("Evaluating : ", img_path)
            print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
            print()

    return score_list

def evaluate_mobilenet(imgs):
    with tf.device('/CPU:0'):
        base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights(weigths_path)

        return evaluate(model,imgs)

def evaluate_nasnet(imgs):
    #needs 224x224 images
    with tf.device('/CPU:0'):
        base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/nasnet_weights.h5')

        return evaluate(model,imgs)

def evaluate_inceptionnet(imgs):
    with tf.device('/CPU:0'):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)
        model.load_weights('weights/inception_resnet_weights.h5')

        return evaluate(model,imgs)


if __name__ == __main__:
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')

    parser.add_argument('-img', type=str, default=[None], nargs='+',
                        help='Pass one or more image paths to evaluate them')

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
        imgs = Path(args.dir).files('[!.]*.png')
        imgs += Path(args.dir).files('[!.]*.jpg')
        imgs += Path(args.dir).files('[!.]*.jpeg')

        for img_path in imgs:
            print(img_path)

    elif args.img[0] is not None:
        print("Loading images from path(s) : ", args.img)
        imgs = args.img

    else:
        raise RuntimeError('Either -dir or -img arguments must be passed as argument')

    score_list = evaluate_mobilenet(imgs)
    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))

                if rename_files:
                    old_file = os.path.join(args.dir, name)
                    new_file = os.path.join(args.dir, "{}_{}".format(i+1,name))
                    os.rename(old_file, new_file)