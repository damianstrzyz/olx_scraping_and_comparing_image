# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--./szukanie_olx/dane/sample.png", help = "path to the image")
args = vars(ap.parse_args())
# load the image
image = cv2.imread("./szukanie_olx/opencv_images/ex2.png")

# define the list of boundaries
boundaries = [([0, 150, 180], [140, 255, 255])]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
	
def confidence(img, template):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    conf = res.max()
    return np.where(res == conf), conf

files = [output]

template = cv2.imread("./szukanie_olx/dane/sample.png")
h, w, _ = template.shape

for name in files:
    img = cv2.imread(name)
    ([y], [x]), conf = confidence(img, template)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text = f'Confidence: {round(float(conf), 2)}'
    cv2.putText(img, text, (x, y), 1, cv2.FONT_HERSHEY_PLAIN, (0, 0, 0), 2)
    cv2.imshow(name, img)
    print (text)
cv2.imshow('Template', template)
cv2.waitKey(0)
