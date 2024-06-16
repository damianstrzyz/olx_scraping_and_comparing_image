
import cv2
import numpy as np

def confidence(img, template):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    conf = res.max()
    return np.where(res == conf), conf

files = ["./szukanie_olx/opencv_images/ex7.png", "./szukanie_olx/opencv_images/ex8.png", "./szukanie_olx/opencv_images/ex9.png", "./szukanie_olx/opencv_images/ex10.png", "./szukanie_olx/opencv_images/ex11.png", "./szukanie_olx/opencv_images/ex12.png", "./szukanie_olx/opencv_images/ex13.png", "./szukanie_olx/opencv_images/ex14.png", "./szukanie_olx/opencv_images/ex15.png"]

template = cv2.imread("./szukanie_olx/dane/sampley.png")
h, w, _ = template.shape

for name in files:
    img = cv2.imread(name)
    ([y], [x]), conf = confidence(img, template)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text = name[-8:] + f' : {round(float(conf), 2)}'
    # cv2.putText(img, text, (x, y), 1, cv2.FONT_HERSHEY_PLAIN, (0, 0, 0), 2)
    # cv2.imshow(name, img)
    print (text)
# cv2.imshow('Template', template)
# cv2.waitKey(0)
