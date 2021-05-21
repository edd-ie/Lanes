import cv2
import numpy as np
import matplotlib.pyplot as plt

# Displaying image
image_c = cv2.imread('test.jpg')
cv2.imshow('given', image_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(image_c)

# Convert coloured to gray
image_g = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

# Perform canny edge detection
image_canny = cv2.Canny(image_g, 50, 200, apertureSize=3)
print(image_canny)

# Visualize edge detection
cv2.imshow('canny', image_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hough line transform to detect lines
lines = cv2.HoughLines(image_canny, 1, np.pi / 180, 300)
print(lines)
print(len(lines))  # Number of line"
print(lines.shape)  # Number of lines

# Plotting straight lines on the image
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)

        a = np.cos(theta)
        b = np.sin(theta)

        # conversion of rho & theta to coordinate (x1, y1,) (x2, y2)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        # Plot
        cv2.line(image_c, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('Hough line', image_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
