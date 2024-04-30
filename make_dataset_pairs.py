import os
import csv
from pathlib import Path

# Define the paths to the image and mask directories
image_dir = Path('datasets/mastectomy/images')
mask_dir = Path('datasets/mastectomy/masks')

# Prepare the CSV file to write the pairs
output_csv = 'datasets/mastectomy/images/train.csv'

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Image' 'Mask'])

    # List all image files in the image directory
    images = sorted(image_dir.glob('*.png'))

    # Iterate through each image file
    for image_path in images:
        # Construct the corresponding mask file path
        mask_path = mask_dir / image_path.name  # Assuming mask and image files have the same name

        # Check if the mask file exists
        if mask_path.exists():
            # Write the image and mask path to the CSV file
            writer.writerow([image_path, mask_path])
        else:
            print(f'Mask for {image_path.name} not found')
