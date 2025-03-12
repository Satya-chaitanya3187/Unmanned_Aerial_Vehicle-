import cv2

# Define class labels
class_labels= { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Load pre-trained model
model_path = r"D:\Machine learning\MobileNetSSD_deploy.prototxt"
weights_path = r"D:\Machine learning\MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNet(model_path, weights_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to match input size of the model
    img_resized = cv2.resize(frame, (300, 300))

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)

    # Forward pass through the network
    detections = net.forward()
    final = detections.squeeze()

    # Display detections on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, _ = frame.shape
    for i in range(final.shape[0]):
        conf = final[i, 2]
        if conf > 0.5:
            class_name = class_labels[final[i, 1]]
            x1n, y1n, x2n, y2n = final[i, 3:]
            x1 = int(x1n * width)
            x2 = int(x2n * width)
            y1 = int(y1n * height)
            y2 = int(y2n * height)
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
            frame = cv2.putText(frame, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
















