from scipy.io import loadmat

img = "/home/aiserver/code/dl_eye/landmark/300W_LP/AFW_Flip/AFW_134212_1_0.jpg"

label = "/home/aiserver/code/dl_eye/landmark/300W_LP/AFW_Flip/AFW_134212_1_0.mat"

m = loadmat(label)
print(m["pt2d"].shape)

import cv2

im_data = cv2.imread(img)

landmark = m["pt2d"]

print(m)

for i in range(68):
    cv2.circle(im_data, (int(landmark[0][i]), int(landmark[1][i])), 2, (0, 255, 0), 2)

cv2.imshow("img", im_data)
cv2.waitKey(0)

