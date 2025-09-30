import numpy as np
import cv2

# Define color target HSV values and tolerance
colors = {
    'red': ([160, 100, 100]),
    'green': ([50, 150, 150]),
    'blue': ([110, 150, 150]),
    'yellow': ([30, 200, 200]),
    'orange': ([15, 150, 150]),
    'purple': ([140, 100, 100]),
    'cyan': ([90, 150, 150]),
    'magenta': ([150, 100, 150]),
    'pink': ([170, 100, 150]),
    'light_blue': ([100, 150, 200]),
    'lime': ([60, 200, 150]),
    'dark_blue': ([120, 150, 100]),
    'brown': ([20, 100, 100]),
    'grey': ([0, 0, 120]),
    'gold': ([25, 200, 150]),
    'violet': ([145, 100, 150]),
    'black': ([0, 0, 10]),
}

# Tolerance for color matching
tolerance = np.array([10, 50, 50])  # Tolerance for hue, saturation, and value

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Load pre-trained face detector (Haar cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

def detect_colors(imageFrame, colors, combined_mask):
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), "uint8")

    for color_name, target_hsv in colors.items():
        lower_np = np.array(target_hsv) - tolerance
        upper_np = np.array(target_hsv) + tolerance
        mask = cv2.inRange(hsvFrame, lower_np, upper_np)
        mask = cv2.dilate(mask, kernel)

        # Apply combined mask to filter out background and face regions
        mask = cv2.bitwise_and(mask, combined_mask)

        # Contours to detect shape and position of the color
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(imageFrame, f"{color_name.capitalize()}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

# Main loop to process video frames
while True:
    ret, imageFrame = webcam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Background subtraction mask
    fg_mask = bg_subtractor.apply(imageFrame)

    # Detect faces
    gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_mask = np.ones_like(fg_mask)

    for (x, y, w, h) in faces:
        face_mask[y:y+h, x:x+w] = 0  # Set face regions to 0 (ignore)

    # Combine foreground mask with face mask
    combined_mask = cv2.bitwise_and(face_mask, fg_mask)

    # Detect colors on objects only
    detect_colors(imageFrame, colors, combined_mask)

    # Display the video with detected colors
    cv2.imshow("Object Color Detection", imageFrame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
