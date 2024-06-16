# import the necessary packages
import numpy as np
import cv2

# Directly assign the image path
image_path = "./szukanie_olx/opencv_images/ex1.png"
image = cv2.imread(image_path)
# Sample image path
template_path = "./szukanie_olx/dane/sampley.png"
template = cv2.imread(template_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Could not load image from path {image_path}")
    exit()

# define the list of boundaries
boundaries = [([0, 150, 180], [140, 255, 255])]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    
    # convert the masked area to yellow
    yellow = np.zeros_like(output)
    yellow[:, :] = [0, 255, 255]  # Yellow color in BGR format
    output_yellow = np.where(mask[:, :, None] != 0, yellow, output)
    
    # show the images
    # cv2.imshow("images", np.hstack([output_yellow]))
    # cv2.waitKey(0)
    #cv2.imwrite("./szukanie_olx/opencv_images/test9.png", output_yellow)

# compare sample and image
def confidence(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    conf = res.max()
    return np.where(res == conf), conf

if template is None:
    print(f"Error: Could not load template from path {template_path}")
    exit()

h, w = template.shape[:2]

# Use the output_yellow variable from the loop
img = output_yellow.copy()
([y], [x]), conf = confidence(img, template)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
text = f'Confidence: {round(float(conf), 2)}'
cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

# Uncomment these lines if you want to display the images
# cv2.imshow("Result", img)
# cv2.imshow("Template", template)
# cv2.waitKey(0)

print(text)
# Uncomment these lines if you want to save the result image
#cv2.imwrite("./szukanie_olx/opencv_images/result.png", img)
