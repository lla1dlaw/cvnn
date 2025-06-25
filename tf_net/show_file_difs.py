import difflib

def compare_files_and_show_diff(file1_path, file2_path):
    """
    Compares two files line by line and prints the differences in a unified diff format.
    """
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            file1_lines = file1.readlines()
            file2_lines = file2.readlines()

        # Generate the unified diff
        diff = difflib.unified_diff(
            file1_lines, file2_lines,
            fromfile=file1_path, tofile=file2_path,
            lineterm='' # Important to avoid extra newlines if lines already contain them
        )

        # Print the differences
        print(f"Differences between {file1_path} and {file2_path}:")
        for line in diff:
            print(line, end='') # Print without adding extra newlines

    except FileNotFoundError:
        print("Error: One or both files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

