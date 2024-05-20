# Path to the input file
input_file_path = 'datasets/mastectomy/images/test.csv'
# Path to the output file
output_file_path = 'datasets/mastectomy/test.csv'

def replace_backslashes(input_file_path, output_file_path):
    # Open the input file in read mode and the output file in write mode
    with open(input_file_path, 'r') as file_read, open(output_file_path, 'w') as file_write:
        # Read each line in the input file
        for line in file_read:
            # Replace backslashes with forward slashes
            modified_line = line.replace('\\', '/')
            # Write the modified line to the output file
            file_write.write(modified_line)

# Call the function with the specified file paths
replace_backslashes(input_file_path, output_file_path)

print("Backslashes have been replaced with forward slashes in the output file.")
