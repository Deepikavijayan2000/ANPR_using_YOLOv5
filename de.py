import cv2
import torch
import numpy as np
import pytesseract
from PIL import Image
import imagehash
import time

model_path = r"\Users\vijay\Downloads\best.pt"  # Custom model path
video_path = r"\Users\vijay\Downloads\project\anpr_video.mp4"
output_video_path = r"\Users\vijay\Downloads\project\output_anpr.mp4"  # Output video path

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Video Input and Output
cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# Text Settings
text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25

# Loop through frames
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    start_time = time.time()  # Track processing time for speed optimization

    # Model Inference
    output = model(image)
    result = np.array(output.pandas().xyxy[0])  # Bounding box results

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
        if license_plate_crop.size == 0:  # Skip empty regions
            continue

        # Skip duplicate plates using seen_plates set
        plate_hash = imagehash.average_hash(Image.fromarray(license_plate_crop))
        if plate_hash in seen_plates:
            continue
        seen_plates.add(plate_hash)

        # Preprocess for OCR
        gray_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_crop = cv2.threshold(gray_crop, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run Tesseract OCR
        license_plate_text = pytesseract.image_to_string(
            Image.fromarray(thresh_crop),
            config='--psm 9 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()

        if len(license_plate_text) >= 5:
            # Display OCR result above the bounding box
            cv2.putText(image, license_plate_text, org=text_origin,
                        fontFace=text_font, fontScale=text_font_scale, color=color, thickness=2)
            print(f"Detected license plate text: {license_plate_text}")

    # Write processed frame to output
    out.write(image)

    # Display the processed frame
    cv2.imshow('License Plate Detection', image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Calculate and display processing speed
    end_time = time.time()
    fps = int(1 / (end_time - start_time))
    print(f"FPS: {fps}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
