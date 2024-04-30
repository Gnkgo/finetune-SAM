# Import the necessary libraries
from PIL import Image
import numpy as np
from pathlib import Path

#loop through folder
# Define the source and target directories
source_dir = Path('C:/Users/Joanna Brodbeck/Documents/GitHub/nnUNET/nnUNet_raw/Dataset011_MastectomyScar/labelsTr1')
target_dir = Path('C:/Users/Joanna Brodbeck/Documents/GitHub/nnUNET/nnUnet_raw/Dataset011_MastectomyScar/labelsTr')
target_dir.mkdir(parents=True, exist_ok=True)  # Create the target directory if it doesn't exist

for file in source_dir.glob('*.png'):
    image = Image.open(file)
    image_array = np.array(image)
    print(image_array.shape)  # (512, 512, 3)
    print(image_array.dtype)  # uint8
    print(image_array)

    for i in range(512):
        for j in range(512):
            for k in range(3):
                if image_array[i][j][k] >= 10:
                    image_array[i][j][k] = 1
                else:
                    image_array[i][j][k] = 0




    corrected_img = Image.fromarray(image_array.astype(np.uint8))
    
    # Save in the target directory with the same file name
    corrected_img.save(target_dir / file.name)

# Convert the image to a NumPy array

