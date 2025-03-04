import cv2
import torch
import numpy as np
import time
import pytesseract
from PIL import Image
import imagehash

model_path = r"\Users\vijay\Downloads\best.pt"  # Custom model path
video_path = r"\Users\vijay\Downloads\project\anpr_video.mp4"

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Video Input and Output
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
output_size = (int(frame_width * 1.5), int(frame_height * 1.5))

# Text and Display Settings
text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25
fps_list = []  # Stores FPS values for removing duplicates
prev_frame_time = time.time()  # Initialize outside the loop
# Loop through frames
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Model Inference
    output = model(image)
    result = np.array(output.pandas().xyxy[0])  # Get bounding box results

    # Process Each Detection
    seen_plates = set()  # Track processed plates to remove duplicates
    for i in result:
        p1 = (int(i[0]), int(i[1]))  # Top-left corner
        p2 = (int(i[2]), int(i[3]))  # Bottom-right corner
        text_origin = (int(i[0]), int(i[1]) - 5)

        # Draw Bounding Boxes
        cv2.rectangle(image, p1, p2, color=color, thickness=2)

        # Crop license plate region
        license_plate_crop = image[p1[1]:p2[1], p1[0]:p2[0]]

        # Skip duplicate plates using seen_plates set
        plate_hash = imagehash.average_hash(Image.fromarray(license_plate_crop))
        if plate_hash in seen_plates:
            continue
        seen_plates.add(plate_hash)

        # Resize and Zoom into the License Plate Crop
        zoomed_crop = cv2.resize(license_plate_crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Apply Preprocessing for OCR
        gray_crop = cv2.cvtColor(zoomed_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_crop = cv2.threshold(gray_crop, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised_image = cv2.bilateralFilter(thresh_crop, d=9, sigmaColor=75, sigmaSpace=75)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

        # Run Tesseract OCR on the preprocessed license plate
        license_plate_text = pytesseract.image_to_string(
            Image.fromarray(sharpened_image),
            config='--psm 9 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()

        if len(license_plate_text) >= 5:
            # Display OCR result above the bounding box
            cv2.putText(image, license_plate_text, org=text_origin,
                        fontFace=text_font, fontScale=text_font_scale, color=color, thickness=2)
            print(f"Detected license plate text: {license_plate_text}")

    # FPS Calculation
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    fps_list.append(fps)

    # Remove duplicate frames based on FPS
    if len(fps_list) > 10:  # Adjust threshold as needed
        if fps_list[-1] == fps_list[-2]:
            continue

    cv2.putText(image, str(fps), (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Resize the image to output size for better visualization
    resized_image = cv2.resize(image, output_size)

    # Display the frame in a window
    cv2.imshow('License Plate Detection', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()