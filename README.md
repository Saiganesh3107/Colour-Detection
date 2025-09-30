# 🎨 Real-Time Colour Detection using OpenCV

This project detects and highlights specific colours in real-time from a webcam feed using **OpenCV**.
It also integrates **face detection** and **background subtraction** to focus colour detection only on objects while ignoring faces and background noise.

---

## ✨ Features

* Detects multiple predefined colours (red, green, blue, yellow, orange, purple, cyan, magenta, pink, light blue, lime, dark blue, brown, grey, gold, violet, black).
* Real-time webcam feed processing.
* **Background subtraction** to filter out unnecessary regions.
* **Face detection** to ignore human faces from colour detection.
* Draws bounding boxes and labels around detected colours.

---

## 🛠️ Requirements

Install the following dependencies before running the project:

```bash
pip install opencv-python numpy
```

---

## 🚀 How to Run

1. Clone or download the repository.
2. Save the script as `colour_detection.py`.
3. Run the script:

   ```bash
   colour_detection.py
   ```
4. A webcam window will open showing detected colours with bounding boxes and labels.
5. Press **`q`** to exit the program.

---

## ⚙️ How It Works

1. **Colour Detection**

   * Each target colour is defined in HSV format with a tolerance range.
   * Contours are drawn around detected colour regions.

2. **Face Detection**

   * Uses Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect faces.
   * Faces are ignored from colour detection.

3. **Background Subtraction**

   * Uses `cv2.createBackgroundSubtractorMOG2()` to remove static background elements.
   * Ensures colour detection works only on moving objects.

---

## 📷 Demo Output

* Bounding boxes appear around objects with the detected colour name.
* Example: If a **red object** is shown to the camera, it will be highlighted with a box labeled **"Red"**.

---

## 📝 Customization

* To add or modify colours, update the `colors` dictionary in the script:

  ```python
  colors = {
      'red': ([160, 100, 100]),
      'green': ([50, 150, 150]),
      # Add your custom colours here
  }
  ```
* Adjust the `tolerance` variable to fine-tune detection sensitivity:

  ```python
  tolerance = np.array([10, 50, 50])
  ```

---

## 📌 Notes

* Ensure your webcam is connected and accessible.
* Performance may vary depending on lighting conditions.
* If detection is inaccurate, adjust HSV values and tolerance.

---


