import os
import csv
import random
from pathlib import Path

# Define the paths to the image and mask directories
image_dir = Path('datasets/mastectomy/images')
mask_dir = Path('datasets/mastectomy/masks')

# Prepare the CSV files to write the pairs
train_csv = 'datasets/mastectomy/train.csv'
val_csv = 'datasets/mastectomy/val.csv'
test_csv = 'datasets/mastectomy/test.csv'

# List all image files in the image directory
images = sorted(image_dir.glob('*.png'))

# Shuffle the list of images
random.shuffle(images)

# Calculate the number of images for each set
total_images = len(images)
train_count = int(0.8 * total_images)
val_count = int(0.15 * total_images)
test_count = total_images - train_count - val_count

# Assign images to training, validation, and testing sets
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# Function to write CSV file
def write_csv(csv_file, image_paths):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        #writer.writerow(['Image', 'Mask'])

        # Write the paths to the CSV file
        for image_path in image_paths:
            # Construct the corresponding mask file path
            mask_path = mask_dir / image_path.name  # Assuming mask and image files have the same name

            if mask_path.exists():
                # Format paths to exclude the 'datasets/' prefix
                formatted_image_path = str(image_path).replace('datasets/', '')
                formatted_mask_path = str(mask_path).replace('datasets/', '')
                writer.writerow([formatted_image_path, formatted_mask_path])
            else:
                print(f'Mask for {image_path.name} not found')

# Write to CSV files
write_csv(train_csv, train_images)
write_csv(val_csv, val_images)
write_csv(test_csv, test_images)
