import os

def combine_txt_files(input_directory, output_file):
    print("Starting the combine_txt_files function")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.isdir(input_directory):
        print(f"Error: The directory '{input_directory}' does not exist.")
        return

    try:
        directory_contents = os.listdir(input_directory)
        print(f"Contents of '{input_directory}': {directory_contents}")
    except Exception as e:
        print(f"Error accessing directory '{input_directory}': {e}")
        return

    try:
        with open(output_file, 'w') as outfile:
            for filename in directory_contents:
                if filename.endswith('.txt'):
                    file_path = os.path.join(input_directory, filename)
                    try:
                        with open(file_path, 'r') as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")  # Add a newline to separate contents of different files
                    except Exception as e:
                        print(f"Error reading file '{file_path}': {e}")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")

input_directory = 'D:/coding/vector/ori_converting_scl_codes'
output_file = 'combined_scl_codes.txt'

combine_txt_files(input_directory, output_file)