import json
import os
from collections import defaultdict

def process_json(input_data):
    """
    Restructures the JSON data, including section_name and filename,
    removing page numbers and grouping by identified_as.

    Args:
        input_data (dict): A dictionary representing the input JSON structure
                             (from _structured.json).

    Returns:
        dict: A dictionary representing the restructured and cleaned JSON.
    """
    # Start the output dictionary with the metadata
    output = {
        "section_name": input_data.get("section_name", "Unknown Section"), # Use .get for safety
        "filename": input_data.get("filename", "Unknown Filename")      # Use .get for safety
    }

    # Create a dictionary to group entries by 'identified_as'
    grouped_data = defaultdict(list)

    # Check if 'content' exists and is a list before iterating
    if isinstance(input_data.get('content'), list):
        # Process each content item
        for item in input_data['content']:
            # Ensure item is a dictionary and has the required keys
            if isinstance(item, dict) and 'identified_as' in item and 'page_number' in item and 'text' in item:
                key = item['identified_as']
                grouped_data[key].append({
                    'page_number': item['page_number'],
                    'text': item['text']
                })
            else:
                print(f"Warning: Skipping invalid item in content: {item}")
    else:
        print(f"Warning: 'content' key is missing or not a list in input data for file {input_data.get('filename')}")


    # Sort and format the grouped data and add it to the output
    for section, entries in grouped_data.items():
        # Sort entries by page number
        sorted_entries = sorted(entries, key=lambda x: x['page_number'])
        # Remove page numbers and keep only text, add to the main output dict
        output[section] = [{'text': entry['text']} for entry in sorted_entries]

    return output

def process_json_file(input_file, output_dir):
    """
    Processes a single JSON file, restructures the data, and saves it to a new JSON file
    in a directory named after the parent directory of the input file in the output directory.

    Args:
        input_file (str): The path to the input JSON file.
        output_dir (str): The main output directory.
    """
    try:
        # Load input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from: {input_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading {input_file}: {e}")
        return

    # Process the data
    try:
        result = process_json(input_data) # Call the updated process_json
    except Exception as e:
        print(f"Error processing JSON data from {input_file}: {e}")
        return

    # Extract the parent directory name (which corresponds to the PDF filename)
    parent_dir_name = os.path.basename(os.path.dirname(input_file))

    # Create a safe directory name for the output
    safe_dirname = parent_dir_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
    output_pdf_dir = os.path.join(output_dir, safe_dirname)  # Output directory for this specific PDF/Company

    # Ensure the output directory exists
    try:
        os.makedirs(output_pdf_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_pdf_dir}: {e}")
        return

    # Extract original filename part (e.g., "Corporate_Overview_structured")
    json_filename_base = os.path.splitext(os.path.basename(input_file))[0]
    # Create the output filename (e.g., "Corporate_Overview_structured_cleaned.json")
    output_filename = os.path.join(output_pdf_dir, f"{json_filename_base}_cleaned.json")


    # Save output
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Processed JSON saved at: {output_filename}")
    except IOError as e:
        print(f"Error saving file {output_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving {output_filename}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    INPUT_DIR = 'pipeline/Scraped_Json'
    OUTPUT_DIR = 'pipeline/Cleaned_Json'

    print(f"Starting JSON cleaning process.")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # Ensure the main output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_files_count = 0
    # Iterate through all directories in the input directory (these are the PDF/Company specific folders)
    for dirname in os.listdir(INPUT_DIR):
        input_company_dir = os.path.join(INPUT_DIR, dirname) # Full path to the company's scraped JSON directory
        # Only process directories
        if os.path.isdir(input_company_dir):
            print(f"\nProcessing directory: {input_company_dir}")
            # Iterate through all JSON files within this company's directory
            for filename in os.listdir(input_company_dir):
                # Target the specific structured JSON files
                if filename.endswith("_structured.json"):
                    input_file = os.path.join(input_company_dir, filename)
                    print(f"  Processing file: {filename}")
                    process_json_file(input_file, OUTPUT_DIR)
                    processed_files_count += 1

    print(f"\nFinished processing JSON files. Total files processed: {processed_files_count}")