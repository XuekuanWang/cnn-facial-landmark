import tensorflow as tf
import glob
import cv2
import numpy as np

pb_path = "saved_model/landmark_model-ResNeXt.ckpt11.pb"

# "resnet_11_0_3_96-color-20181205-1228-1206-10-stp2-10-10.pb"
input = "Placeholder:0"  ##'finger_net/fcn/deconv_final/BiasAdd:0'
output = "logits/BiasAdd:0"

session = tf.Session()

with session.as_default():
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = session.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

im_list = glob.glob("/home/aiserver/muke/cnn-facial-landmark/image/*")
landmark = session.graph.get_tensor_by_name(output)
idxx = 0
for im_path in im_list:
    im_data = cv2.imread(im_path)
    im_data = cv2.resize(im_data, (128, 128))
    predictions = session.run(landmark,
                               {input: np.expand_dims(im_data, 0)})
    print(np.expand_dims(im_data, 0).shape)
    print(predictions)
    predictions = predictions[0]
    for i in range(0, 136, 2):
        cv2.circle(im_data, (int(predictions[i]),
                                 int(predictions[i + 1])),
                       2, (0, 255, 0), 2)

    cv2.imwrite("res/{}.jpg".format(idxx), im_data)
    idxx += 1
    cv2.imshow("image", im_data)
    cv2.waitKey(0)