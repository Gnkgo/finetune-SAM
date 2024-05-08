import os
import csv
from pathlib import Path

# Define the paths to the image and mask directories
image_dir = Path('datasets/mastectomy/images')

# Prepare the CSV file to write the pairs
output_csv = 'datasets/mastectomy/images/test.csv'

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    #writer.writerow(['Image' 'Mask'])

    # List all image files in the image directory
    images = sorted(image_dir.glob('*.png'))

    # Iterate through each image file
    for image_path in images:
        # Construct the corresponding mask file path
        if '0000' in image_path.name:
            # Format paths to exclude the 'datasets/' prefix
            formatted_image_path = str(image_path).replace('datasets/', '')
            print(f'Image: {formatted_image_path}')

            # Write the formatted image and mask path to the CSV file
            writer.writerow([formatted_image_path])
        else:
            print(f'Mask for {image_path.name} not found')
