import cv2
from find_elements import hex_to_hsv_range
import os
# Load the image
output_dir = "./arina/"  # Replace with your desired output path

def del_bg(input_path, output_path):
    image = cv2.imread(input_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower, upper, l2, u2 = hex_to_hsv_range('#EB99B9')

    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower, upper)

    # Invert the mask to keep the foreground
    foreground = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))

    # Convert to BGRA to add an alpha channel
    rgba = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)

    # Set transparent where the mask is green
    rgba[:, :, 3] = cv2.bitwise_not(mask)

    # Save the output
    cv2.imwrite(output_path, rgba)


input_path = "./bg"
files = os.listdir(input_path)

# Iterate through files
for file in files:
    file_input_path = os.path.join(input_path, file)
    file_save_path = os.path.join(output_dir, file)
    if os.path.isfile(file_input_path):  # Check if it's a file (not a folder)
        print(f"Reading file: {file_input_path}")
        del_bg(file_input_path, file_save_path.replace(".jpg", ".png"))

