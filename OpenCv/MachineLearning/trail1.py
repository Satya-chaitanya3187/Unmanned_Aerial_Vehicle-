import cv2
import matplotlib.pyplot as plt

# Load image
image_path = "Photo.jpg"
image = cv2.imread(image_path)

# Display image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis labels
plt.show()
