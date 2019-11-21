import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/data_2/try/out_nanjie_img/0_0_吉_0.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gay, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



cv2.drawContours(gay, contours, -1, (0, 255, 255), 2)


plt.figure('original image with contours'), plt.imshow(gay, cmap = 'gray')
plt.show()
