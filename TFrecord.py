import tensorflow as tf
import os
import numpy as np
import random
from glob import glob


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # 字节列表不会从EagerTensor中解压一个字符串
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# noinspection PyRedeclaration
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, image_name, label):
    image_shape = tf.image.decode_png(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image': _bytes_feature(image_string),
        'image_name': _bytes_feature(image_name),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def bytes_to_file(file, image_bytes):
    with open(file, 'wb') as f:
        f.write(image_bytes)
        return file


def BuildTFRecord(ImageSetsPath, TFRecordSetsPath, Label, ImageFormat):
    ImageSets = os.listdir(ImageSetsPath)
    CompImageSetsPaths = []
    for i in ImageSets:
        I = ImageSetsPath + "/" + i
        CompImageSetsPaths.append(I)

    number = 1
    number_all = len(CompImageSetsPaths)
    TFRecords = os.listdir(TFRecordSetsPath)
    for i in CompImageSetsPaths:
        print(number, "/", number_all, i.split("/")[-1])
        number += 1
        RecordFile = TFRecordSetsPath + "/" + i.split("/")[-1] + ".tfrecords"
        if RecordFile.split("/")[-1] not in TFRecords:
            Images = os.listdir(i)
            CompImagePaths = []
            for o in Images:
                a = i + "/" + o
                CompImagePaths.append(a)
            CompImagePaths = np.array(CompImagePaths)
            with tf.io.TFRecordWriter(RecordFile) as writer:
                for ImagePath in CompImagePaths:
                    image_string = open(ImagePath, 'rb').read()
                    image_name = ImagePath.split("/")[-1].strip("." + ImageFormat)
                    image_name_bytes = bytes(image_name, 'utf-8')
                    tf_example = image_example(image_string, image_name_bytes, Label)
                    writer.write(tf_example.SerializeToString())


def make_data(example_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.io.parse_single_example(example_string, image_feature_description)
    image = parsed_example["image"]
    # print(image)
    # image1 = tf.io.decode_raw(image, out_type=tf.uint8)
    # image2 = tf.io.parse_tensor(image, out_type=tf.uint8)
    image3 = tf.io.decode_png(image, channels=3)
    # print(image1, image2, image3)
    # image_reshape1 = tf.reshape(image1, [224, 224, 3])
    # image_reshape2 = tf.reshape(image2, [224, 224, 3])
    image_reshape3 = tf.image.resize(image3, [224, 224])
    # print(image_reshape1, image_reshape2, image_reshape3)
    label = parsed_example["label"]
    name = parsed_example["image_name"]
    # print(label)

    return image_reshape3, label, name



def main():
    data_root_dir = "H:/NTRK/PanCancer/STAIN_Patches"
    tfrecord_root_dir = "H:/NTRK/PanCancer/STAIN_Tfrecords"

    Pos_data_dir = data_root_dir + "/" + "1"
    Neg_data_dir = data_root_dir + "/" + "0"
    Pos_tfrecord_dir = tfrecord_root_dir + "/" + "1"
    Neg_tfrecord_dir = tfrecord_root_dir + "/" + "0"

    BuildTFRecord(Pos_data_dir, Pos_tfrecord_dir, 1, "png")
    BuildTFRecord(Neg_data_dir, Neg_tfrecord_dir, 0, "png")


def main1():
    tfrecord_root_dir = "H:/NTRK/PanCancer/STAIN_Tfrecords"
    tfrecords_dir_0 = tfrecord_root_dir + "/" + "0"
    tfrecords_dir_1 = tfrecord_root_dir + "/" + "1"

    tfrecords_0 = [os.path.join(tfrecords_dir_0, i) for i in os.listdir(tfrecords_dir_0)]
    tfrecords_1 = [os.path.join(tfrecords_dir_1, i) for i in os.listdir(tfrecords_dir_1)]
    tfrecords = tfrecords_0 + tfrecords_1
    random.seed(1225)
    random.shuffle(tfrecords)
    image_dataset = tf.data.TFRecordDataset(tfrecords)
    shuffle_image_dataset = image_dataset.shuffle(buffer_size=80000, reshuffle_each_iteration=False, seed=1225)
    parsed_image_dataset = shuffle_image_dataset.map(make_data)
    dataset = parsed_image_dataset.batch(32).prefetch(1)
    for i in dataset:
        print(i)
    return dataset


def main2():
    tfrecord_root_dir = "H:/NTRK/PanCancer/STAIN_Tfrecords"
    tfrecords_dir = tfrecord_root_dir + "/" + "1"
    tfrecords = [os.path.join(tfrecords_dir, i) for i in os.listdir(tfrecords_dir)]

    # tfrecord = ["H:/NTRK/PanCancer/STAIN_Tfrecords/0/001-100055ASA1L1.tfrecords", "H:/NTRK/PanCancer/STAIN_Tfrecords/1/001-103832ASA1S1.tfrecords"]
    image_dataset = tf.data.TFRecordDataset(tfrecords)
    shuffle_image_dataset = image_dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=False, seed=123)
    parsed_image_dataset = shuffle_image_dataset.map(make_data)


if __name__ == "__main__":
    main1()
