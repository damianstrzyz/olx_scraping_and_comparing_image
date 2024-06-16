import cv2
import matplotlib.pyplot as plt

# Read images
examples = [cv2.imread(img) for img in ['./szukanie_olx/opencv_images/ex1.png', './szukanie_olx/opencv_images/ex2.png', './szukanie_olx/opencv_images/ex3.png', './szukanie_olx/opencv_images/ex4.png', './szukanie_olx/opencv_images/ex5.png']]
target = cv2.imread('./szukanie_olx/dane/sample.png')
h, w = target.shape[:2]

# Iterate examples
for i, img in enumerate(examples):

    # Template matching
    # cf. https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html
    res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)

    # Get location of maximum
    _, max_val, _, top_left = cv2.minMaxLoc(res)

    # Set up threshold for decision target found or not
    thr = 0.7
    if max_val > thr:

        # Show found target in example
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Visualization
    plt.figure(i, figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img[..., [2, 1, 0]]), plt.title('Example')
    plt.subplot(1, 2, 2), plt.imshow(res, vmin=0, vmax=1, cmap='gray')
    plt.title('Matching result'), plt.colorbar(), plt.tight_layout()

plt.show()
