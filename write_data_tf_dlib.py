import tensorflow as tf
import cv2
import numpy as np
import glob

im_list = glob.glob("/home/aiserver/muke/dataset/landmark-data/image/*")

tfrecord_file_train = "data/train.tfrecord"
tfrecord_file_test = "data/test.tfrecord"

im_size = 128

def write_data(begin, end, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    index = [i for i in range(im_list.__len__())]
    np.random.shuffle(index)
    for i in range(begin, end):
        im_d = im_list[index[i]]
        im_l = im_list[index[i]].replace("/image/", "/label/").replace("jpg", "txt")
        print(i, im_d)
        data = cv2.imread(im_d)
        sp = data.shape

        im_point = []
        for p in open(im_l).readlines():
            p = p.replace("\n", "").split(",")
            im_point.append(int(int(p[0]) * im_size / sp[1]))
            im_point.append(int(int(p[1]) * im_size / sp[0]))

        data = cv2.resize(data, (im_size, im_size))

        #data = tf.gfile.FastGFile(im_d, "rb").read()
        ex = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "image":tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[data.tobytes()])),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=im_point)),
                }
            )
        )
        writer.write(ex.SerializeToString())

    writer.close()

write_data(0, int(im_list.__len__() * 0.9), tfrecord_file_train)
write_data(int(im_list.__len__() * 0.9), im_list.__len__(), tfrecord_file_test)