import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

# Load pre-trained depth estimation model
model = resnet18(pretrained=True)
model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict depth from image
def predict_depth(image):
    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Predict depth
    with torch.no_grad():
        output = model(input_batch)
        depth_map = output.squeeze().cpu().numpy()

    return depth_map

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize frame to match input size of the model
    frame_resized = cv2.resize(frame, (224, 224))

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Predict depth
    try:
        depth_map = predict_depth(frame_rgb)

        # Normalize depth map for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display original frame
        cv2.imshow('Original', frame)

        # Display depth map (if not empty)
        if depth_map_normalized is not None:
            # Resize depth map for better visualization
            depth_map_resized = cv2.resize(depth_map_normalized, (frame.shape[1], frame.shape[0]))

            # Display resized depth map
            cv2.imshow('Depth Map', depth_map_resized)

    except Exception as e:
        print("Error:", e)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

