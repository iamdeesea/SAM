import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the screenshot image
img = cv2.imread("juliflora_screenshot.png")  # Change filename if needed
if img is None:
    raise FileNotFoundError("Could not load 'juliflora_screenshot.png' – check path and filename.")

# Convert to HSV to detect green regions
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green range in HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create binary mask for green areas
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: clean up noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# Show the detected mask
plt.imshow(mask, cmap='gray')
plt.title("Detected Green Areas")
plt.show()

# Save the binary mask
cv2.imwrite("detected_mask.png", mask)

print("✅ Saved binary mask as 'detected_mask.png'")
