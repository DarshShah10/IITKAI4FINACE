import fitz  # PyMuPDF
import json
import re
import os
from collections import defaultdict
from thefuzz import fuzz  # pip install PyMuPDF thefuzz python-Levenshtein

# --- Configuration ---
# Make sure this path points to your schema file
SCHEMA_FILE = '/Users/ginger/Developer/techkriti/pipeline/report_schema.json'
# Define where the output structured JSON files will be saved (now an input)
# OUTPUT_DIR = 'structured_report_output_keyword_single_script' # REMOVED: Now passed as an argument

# -- Matching Parameters --
CHARS_TO_CHECK_FOR_HEADING = 800  # How many characters from the start of a page to check
FUZZY_MATCH_THRESHOLD = 88  # Minimum score (0-100) for fuzzy matching schema variations
MIN_HEADING_LENGTH = 5  # Minimum length for a potential heading match to be considered valid

# --- Text Extraction Function ---


def extract_text_from_pdf(pdf_path):
    """Extract plain text from each page of a PDF using pymupdf."""
    text_data = []
    try:
        doc = fitz.open(pdf_path)
        print(f"Opened PDF: {pdf_path} ({len(doc)} pages)")
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                text_data.append(text if text else "")  # Append empty string if page has no text
            except Exception as e_page:
                print(f"Warning: Could not extract text from page {page_num + 1}. Error: {e_page}")
                text_data.append("")  # Append empty string on error
        doc.close()
        print(f"Successfully extracted text from {len(text_data)} pages.")
        return text_data
    except Exception as e_doc:
        print(f"Error opening or processing PDF file {pdf_path}: {e_doc}")
        return None

# --- Helper Functions ---


def load_json(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from - {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        return None


def save_json(data, filepath):
    """Saves data to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved structured data to: {filepath}")
    except IOError as e:
        print(f"Error saving file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving {filepath}: {e}")


def clean_text(text):
    """Basic text cleaning for matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Minimal cleaning for keyword matching
    return text


def build_variation_map(schema):
    """Creates a map from cleaned variation text to (main_section, sub_section) keys."""
    variation_map = {}
    if not schema:
        print("Error: Schema data is missing or invalid.")
        return {}, []

    # Check if schema is a dictionary
    if not isinstance(schema, dict):
        print("Error: Schema is not a dictionary.")
        return {}, []

    print("Building variation map...")
    count = 0
    for main_key, main_data in schema.items():
        if not isinstance(main_data, dict):
            # print(f"Warning: Skipping invalid main section data for key '{main_key}'. Expected dict.")
            continue

        # Add main section variations
        main_variations = main_data.get("variations", [])
        if isinstance(main_variations, list):
            for var in main_variations:
                if isinstance(var, str):
                    cleaned_var = clean_text(var)
                    if len(cleaned_var) >= MIN_HEADING_LENGTH:
                        if cleaned_var not in variation_map:
                            variation_map[cleaned_var] = (main_key, main_key)
                            count += 1
                # else:
                # print(f"Warning: Invalid variation type in '{main_key}': {type(var)}")

        # Add subsection variations
        subsections = main_data.get("subsections", {})
        if isinstance(subsections, dict):
            for sub_key, sub_data in subsections.items():
                if not isinstance(sub_data, dict):
                    # print(f"Warning: Skipping invalid subsection data for key '{main_key} -> {sub_key}'. Expected dict.")
                    continue
                sub_variations = sub_data.get("variations", [])
                if isinstance(sub_variations, list):
                    for var in sub_variations:
                        if isinstance(var, str):
                            cleaned_var = clean_text(var)
                            if len(cleaned_var) >= MIN_HEADING_LENGTH:
                                variation_map[cleaned_var] = (main_key, sub_key)
                                # Only increment count if it's a new variation overall
                                if cleaned_var not in variation_map or variation_map[cleaned_var] == (main_key, main_key):
                                    count += 1
                        # else:
                        # print(f"Warning: Invalid variation type in '{main_key} -> {sub_key}': {type(var)}")
                # else:
                # print(f"Warning: Invalid 'variations' type in '{main_key} -> {sub_key}': {type(sub_variations)}")
        # else:
        # print(f"Warning: Invalid 'subsections' type in '{main_key}': {type(subsections)}")

    # Sort variations by length descending
    sorted_variations = sorted(variation_map.keys(), key=len, reverse=True)
    print(f"Built variation map with {count} unique cleaned variations.")
    if not variation_map:
        print("Warning: Variation map is empty. Check schema format and content.")
        return variation_map, sorted_variations
    return variation_map, sorted_variations


def find_best_match_in_text(page_fragment, variation_map, sorted_variations, threshold):
    """
    Finds the best (highest score, longest variation) fuzzy match for a schema variation
    within the beginning fragment of the page text.
    """
    best_match_info = None
    highest_score = -1
    cleaned_fragment = clean_text(page_fragment)  # Clean the fragment once

    if not cleaned_fragment:
        return None

    for variation in sorted_variations:
        # Optimization: If variation is longer than fragment, partial_ratio is needed
        if len(variation) > len(cleaned_fragment) + 15:  # Allow some leeway
            continue  # Skip if variation is significantly longer

        # Use partial_ratio as heading might be only part of the fragment
        score = fuzz.partial_ratio(cleaned_fragment, variation)

        if score >= threshold and score > highest_score:
            highest_score = score
            match_keys = variation_map.get(variation)  # Use .get for safety
            if match_keys:  # Ensure the variation was actually in the map
                best_match_info = {
                    "main_key": match_keys[0],
                    "sub_key": match_keys[1],
                    "matched_variation": variation,
                    "score": highest_score
                }
                if highest_score == 100:  # Perfect score optimization
                    break

    return best_match_info


# --- Main Structuring Logic ---


def structure_report_data_keyword(page_texts, schema):
    """Structures the report text based purely on keyword matching at the start of pages."""
    if not page_texts:
        print("Error: No text data provided for structuring.")
        return None
    if not schema:
        print("Error: Schema not loaded or invalid.")
        return None

    print("\nStructuring document based on keyword matching...")
    structured_output = defaultdict(list)
    variation_map, sorted_variations = build_variation_map(schema)
    if not variation_map:
        print("Error: Cannot proceed without a valid variation map.")
        # Add all pages to Unclassified if map fails?
        for page_idx, page_text in enumerate(page_texts):
            structured_output["Unclassified"].append({
                "page_number": page_idx + 1,
                "identified_as": "Unclassified (Schema Error)",
                "text": page_text
            })
        return structured_output

    current_main_section = "Unclassified"
    current_sub_section = "Unclassified"

    for page_idx, page_text in enumerate(page_texts):
        page_num = page_idx + 1
        # Handle potential non-string entries in page_texts if extraction had errors
        if not isinstance(page_text, str):
            page_text = ""  # Treat as empty page

        page_fragment = page_text[:CHARS_TO_CHECK_FOR_HEADING]

        best_match = find_best_match_in_text(page_fragment, variation_map, sorted_variations, FUZZY_MATCH_THRESHOLD)

        if best_match and best_match["sub_key"] != current_sub_section:
            match_main = best_match['main_key']
            match_sub = best_match['sub_key']
            match_var = best_match['matched_variation']
            match_score = best_match['score']

            print(f"  Page {page_num}: Found Heading '{match_var}' (Score: {match_score}). Assigning section -> {match_main}::{match_sub}")
            current_main_section = match_main
            current_sub_section = match_sub
        # Else: Continue with the previous section assignment

        page_entry = {
            "page_number": page_num,
            "identified_as": current_sub_section,
            "text": page_text
        }
        structured_output[current_main_section].append(page_entry)

    print("\nProcessing complete.")
    total_pages = 0
    for key, pages in structured_output.items():
        print(f"  - Section '{key}': {len(pages)} pages assigned.")
        total_pages += len(pages)
    print(f"  Total pages processed: {total_pages} / {len(page_texts)}")

    return structured_output


# --- Main Function for Pipeline ---


def process_pdf(pdf_path, output_dir):
    """
    Processes a single PDF, extracts data, structures it, and saves separate JSON
    files for each section in a directory named after the PDF.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): The directory to save the structured JSON files.
    """
    print(f"--- Processing PDF: {pdf_path} ---")

    # 1. Load Schema
    print(f"Loading schema from: {SCHEMA_FILE}")
    report_schema = load_json(SCHEMA_FILE)
    if not report_schema:
        print("Error: Failed to load schema. Exiting.")
        return

    # 2. Extract Text from PDF
    print(f"Extracting text from PDF: {pdf_path}")
    page_texts = extract_text_from_pdf(pdf_path)
    if page_texts is None:
        print("Error: Failed to extract text from PDF. Exiting.")
        return
    if not page_texts:
        print("Warning: No text could be extracted from the PDF.")
        # Optionally exit or proceed to save empty files
        # exit()

    # 3. Structure the Data using Keyword Approach
    structured_data = structure_report_data_keyword(page_texts, report_schema)

    # 4. Save Output Files (One JSON per section)
    if structured_data:
        # Extract filename from path without extension
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        # Create a safe directory name for the output
        safe_dirname = pdf_filename.replace(' ', '_').replace('/', '_').replace('&', 'and')
        output_pdf_dir = os.path.join(output_dir, safe_dirname)  # Changed output dir to have folder for each pdf
        # Ensure the output directory exists
        os.makedirs(output_pdf_dir, exist_ok=True)

        # Define the main sections to save
        main_sections_to_save = [
            "Corporate Overview",
            "Statutory Reports",
            "Financial Statements",
            "Other Information",
            "Unclassified"
        ]

        for section_name in main_sections_to_save:
            # Get content, default to empty list if section not found
            content = structured_data.get(section_name, [])
            # Create safe filename for section
            safe_section_name = section_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
            output_filename = os.path.join(output_pdf_dir, f"{safe_section_name}_structured.json")  # added join path

            save_json(
                {"section_name": section_name, "filename": pdf_filename, "extraction_method": "Keyword/Fuzzy", "content": content},
                output_filename
            )
    else:
        print("No structured data generated.")

    print(f"--- Finished processing {pdf_path} ---")


# --- Script Execution (Pipeline Handling) ---

if __name__ == "__main__":
    print("--- Starting Batch Report Structuring Script ---")

    INPUT_DIR = "/Users/ginger/Developer/techkriti/pipeline/data3"  # Where the PDF files are located
    OUTPUT_DIR = "pipeline/Scraped_Json"  # Where the structured JSON files will be saved
    # INPUT_DIR = input("Enter the input directory: ")
    # OUTPUT_DIR = input("Enter the output directory: ")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate through all PDF files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            process_pdf(pdf_path, OUTPUT_DIR)

    print("\n--- Batch script finished. ---")