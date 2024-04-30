# Import the necessary libraries
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

# Define the source and target directories
source_dir = Path('C:/Users/Joanna Brodbeck/Documents/GitHub/nnUNET/nnUNet_raw/Dataset011_MastectomyScar/labelsTr1')
target_dir = Path('C:/Users/Joanna Brodbeck/Documents/GitHub/nnUNET/nnUNet_raw/Dataset011_MastectomyScar/labelsTr')
target_dir.mkdir(parents=True, exist_ok=True)  # Create the target directory if it doesn't exist

# Iterate through all '.png' files in the source directory
for file in source_dir.glob('*.png'):
    img = Image.open(file)

# Convert to RGB (this drops the alpha channel)
    rgb_img = img.convert('RGB')

    # Save or use the image without the alpha channel
            
    image_array = np.array(rgb_img)


    
   
    # if one of the channel is bigger than 1 set it to 1 for each channel otherwise to 0

    # Normalize pixel values to 0 or 1
    # Set to 1 if the RGB value is closer to white (255, 255, 255), else set to 0
    normalized_array = np.where(image_array >= 15, 1, 0)

    # Convert the normalized array back to an image
    normalized_img = Image.fromarray(normalized_array.astype(np.uint8))
    
    #print(normalized_array.shape)  # (512, 512, 3)
    #print(normalized_img)$
    # Save in the target directory with the same file name
    
    normalized_img.save(target_dir / file.name)

    
    