import os

# The directory where your label images are located
label_directory = "datasets/mastectomy/images"

# Iterate through all files in the directory
for filename in os.listdir(label_directory):
    # Split the filename to isolate the base name and extension
    base_name, extension = os.path.splitext(filename)
    
    # If the base name ends with '_0000', remove it
    if base_name.endswith("_0000"):
        new_base_name = base_name[:-5]  # Removes the last 5 characters ('_0000')
        new_filename = new_base_name + extension  # Reattach the extension
        
        # Full paths for renaming
        old_path = os.path.join(label_directory, filename)
        new_path = os.path.join(label_directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
