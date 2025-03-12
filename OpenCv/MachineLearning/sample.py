import cv2

# Specify the image path
image_path = r"D:\Wallpapers\Archaic.jpg"

# Load an image using imread
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is not None:
    print("Image loaded successfully.")
    print("Image shape:", image.shape)  # Print the dimensions of the loaded image
else:
    print("Failed to load image. Please check the file path.")
