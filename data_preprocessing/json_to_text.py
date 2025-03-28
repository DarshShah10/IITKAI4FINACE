import json
import os

def clean_text(text):
    """Removes excessive new lines, replaces them with full stops, and removes tabs."""
    text = text.replace("\t", " ")  # Remove tabs
    text = text.replace("\n", ". ")  # Replace newlines with periods for readability
    text = text.replace("..", ".")   # Clean up extra dots
    return text.strip()

def format_text(data):
    """Formats the JSON data into structured text."""
    formatted_text = []
    
    # Main section heading
    if "section_name" in data:
        formatted_text.append(f"{data['section_name']}\n" + "=" * len(data['section_name']) + "\n")
    
    for key, value in data.items():
        if key == "section_name":
            continue  # Skip section name as it's already added

        formatted_text.append(f"{key}\n" + "-" * len(key))  # Subheading

        if isinstance(value, str):  # If the value is a string, add it directly
            cleaned_text = clean_text(value)
            formatted_text.append(cleaned_text + "\n")
        elif isinstance(value, list):  # If the value is a list of paragraphs
            for entry in value:
                if isinstance(entry, dict) and "text" in entry:
                    cleaned_text = clean_text(entry["text"])
                    formatted_text.append(cleaned_text + "\n")  # Add formatted paragraph
    
    return "\n".join(formatted_text)


def process_json_file(input_file, output_dir):
    """
    Processes a single JSON file, formats the data, and saves it to a text file
    in a directory named after the parent directory of the input file in the output directory.
    
    Args:
        input_file (str): The path to the input JSON file.
        output_dir (str): The main output directory.
    """
    try:
        # Load input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from: {input_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading {input_file}: {e}")
        return
    
    # Process and format the text
    try:
        formatted_output = format_text(data)
    except Exception as e:
        print(f"Error processing JSON data from {input_file}: {e}")
        return

    # Extract the parent directory name
    parent_dir_name = os.path.basename(os.path.dirname(input_file))
    
    # Create a safe directory name for the output
    safe_dirname = parent_dir_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
    output_pdf_dir = os.path.join(output_dir, safe_dirname)  # Path is folder for file

    # Ensure the output directory exists
    try:
        os.makedirs(output_pdf_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_pdf_dir}: {e}")
        return

    # Extract filename from the input_file
    json_filename = os.path.splitext(os.path.basename(input_file))[0]

    output_filename = os.path.join(output_pdf_dir, f"{json_filename}.txt") # create text file


    # Save output to a text file
    try:
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(formatted_output)
        print(f"Formatted text saved to: {output_filename}")
    except IOError as e:
        print(f"Error saving file {output_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving {output_filename}: {e}")

# Example usage (pipeline)
if __name__ == "__main__":
    INPUT_DIR = 'pipeline/Cleaned_Json'
    OUTPUT_DIR = 'pipeline/Structured_Data'

    # Ensure the main output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate through all directories in the input directory
    for dirname in os.listdir(INPUT_DIR):
        input_pdf_dir = os.path.join(INPUT_DIR, dirname) # get join path
        # Only process directories
        if os.path.isdir(input_pdf_dir): # filter out only folders
            # Iterate through all JSON files in the directory
            for filename in os.listdir(input_pdf_dir):
                if filename.endswith("_cleaned.json"):
                    input_file = os.path.join(input_pdf_dir, filename) # get folder path to save
                    process_json_file(input_file, OUTPUT_DIR)

    print("Finished processing JSON files.")