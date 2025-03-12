import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define class labels
class_labels = {
    1: "person",
    2: "cat",
    3: "dog",
    4: "motorbike",
    5: "sheep",
    6: "bottle"
}

# Load image
image_path = "D:\Machine learning\Photo.jpg"
image = cv2.imread(image_path)

# Load pre-trained model
model_path = "D:\Machine learning\MobileNetSSD_deploy.prototxt"
weights_path = "D:\Machine learning\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNet(weights_path, model_path)

# Prepare input image for segmentation
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (127.5, 127.5, 127.5), False)

# Set input to the network
net.setInput(blob)

# Run forward pass
output = net.forward()

# Process output masks
h, w = image.shape[:2]
for i in range(output.shape[1]):
    # Extract confidence and class ID
    confidence = output[0, i, :, 0]
    class_id = int(output[0, i, 0, 1])

    # Filter detections by confidence threshold
    mask = (confidence > 0.5).astype(np.uint8)

    # Resize mask to match input image size
    mask = cv2.resize(mask, (w, h))

    # Find contours of the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes and labels
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Print bounding box coordinates
        print("Bounding box coordinates:", x, y, w, h)
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get class label
        if class_id in class_labels:
            class_label = class_labels[class_id]
        else:
            class_label = "Unknown"
        
        # Draw class label at the top left corner of the box
        cv2.putText(image, class_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display result using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
