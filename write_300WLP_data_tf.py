import tensorflow as tf
import cv2
import numpy as np
import glob
from scipy.io import loadmat

folder = ["AFW","HELEN","IBUG","LFPW"]

im_list = []
for folder_item in folder:
    im_list += glob.glob("/home/aiserver/code/dl_eye/landmark/300W_LP/landmarks/"
                         + folder_item + '/*.mat')

print(im_list)
index = [i for i in range(im_list.__len__())]
np.random.shuffle(index)

tfrecord_file_train = "data/train.tfrecord"
tfrecord_file_test = "data/test.tfrecord"
im_size = 128

def write_data(begin, end, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for i in range(begin, end):
        im_l = im_list[index[i]]
        im_d = im_list[index[i]].replace("300W_LP/landmarks/", "300W_LP/").replace("_pts.mat", ".jpg")
        print(i, im_d, im_l)
        data = cv2.imread(im_d)
        sp = data.shape
        m = loadmat(im_l)
        #print(m)
        landmark = m["pts_2d"]
        im_point = []
        print(landmark.shape)

        x_max = int(np.max(landmark[0:68, 0]))
        x_min = int(np.min(landmark[0:68, 0]))

        y_max = int(np.max(landmark[0:68, 1]))
        y_min = int(np.min(landmark[0:68, 1]))


        y_min = int(y_min - (y_max - y_min) * 0.3)

        data = data[y_min:y_max, x_min:x_max]

        sp = data.shape

        for p in range(68):
            im_point.append(int((int(landmark[p][0]) - x_min) * im_size / sp[1]))
            im_point.append(int((int(landmark[p][1]) - y_min) * im_size / sp[0]))
            #cv2.circle(data, (int(landmark[i][0]) - x_min, int(landmark[i][1]) - y_min), 2, (0, 255, 0), 2)
        # cv2.imshow("img", data)
        # cv2.waitKey(1)
        data = cv2.resize(data, (im_size, im_size))

        cv2.imwrite("image/{}.jpg".format(i), data)
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