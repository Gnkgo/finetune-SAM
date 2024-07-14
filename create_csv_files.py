import os
import random
import csv

# Define paths
images_path = './datasets/nasare/images'
masks_path = './datasets/nasare/masks'

# Get list of all image filenames (assuming they all end with .png)
image_filenames = [f for f in os.listdir(images_path) if f.endswith('.png')]

# Shuffle the filenames
random.shuffle(image_filenames)

# Split into training and validation sets (80% train, 20% validation)
split_index = int(len(image_filenames) * 0.8)
train_filenames = image_filenames[:split_index]
valid_filenames = image_filenames[split_index:]

# Function to write CSV file
def write_csv(filename, filenames, images_path, masks_path):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for fname in filenames:
            image_path = os.path.join(images_path, fname)
            mask_path = os.path.join(masks_path, fname)
            image_path = image_path.replace('\\', '/')
            mask_path = mask_path.replace('\\', '/')
            image_path = image_path.replace('./datasets/', '')  
            mask_path = mask_path.replace('./datasets/', '')
            writer.writerow([image_path, mask_path])

# Write train.csv
write_csv('train.csv', train_filenames, images_path, masks_path)

# Write valid.csv
write_csv('valid.csv', valid_filenames, images_path, masks_path)

print("CSV files created successfully.")
