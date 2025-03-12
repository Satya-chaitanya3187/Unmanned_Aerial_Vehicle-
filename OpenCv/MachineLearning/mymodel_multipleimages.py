import matplotlib.pyplot as plt
import cv2

def func1(xdic,keyword):
    for i in xdic:
        for y in xdic[i]:
            if y==keyword:
                return i

# Function to perform object detection on an image
def detect_objects(image, net, class_labels,xdic,i):
    # Resize frame to match input size of the model
    img_resized = cv2.resize(image, (300, 300))
    xdic[j-1]=[]

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)

    # Perform forward pass to get detections
    detections = net.forward()
    final = detections.squeeze()

    # Process detections
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, _ = image.shape
    for i in range(final.shape[0]):
        conf = final[i, 2]
        if conf > 0.5:
            xdic[j-1].append(class_labels[final[i,1]])
            class_names = class_labels[final[i, 1]]
            x1n, y1n, x2n, y2n = final[i, 3:]
            x1 = int(x1n * width)
            x2 = int(x2n * width)
            y1 = int(y1n * height)
            y2 = int(y2n * height)
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
            image = cv2.putText(image, class_names, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return image

# Paths to model and class labels
model_path = r"D:\Machine learning\MobileNetSSD_deploy.prototxt"
weights_path = r"D:\Machine learning\MobileNetSSD_deploy.caffemodel"
class_labels= { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Read the model
net = cv2.dnn.readNet(model_path, weights_path)

# List of image paths
image_paths = ["Photo.jpg","Photo1.jpg"]  # Add more image paths as needed
n_images=len(image_paths)
xdic={}

# Process each image
j=0
result_image=[]
for image_path in image_paths:
    j=j+1
    # Read the image
    image = cv2.imread(image_path)
    result_image.append(detect_objects(image, net, class_labels,xdic,j))
keyword=input("Enter the keyword")
index=func1(xdic,keyword)
plt.imshow(result_image[index])
plt.show()