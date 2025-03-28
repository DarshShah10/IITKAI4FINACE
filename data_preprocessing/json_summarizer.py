import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# IMPORTANT: Use raw string (r"...") or double backslashes (\\) for Windows paths
INPUT_DIR = r"C:\Darsh\Techkriti_Finance\Cleaned_Json-20250327T210137Z-001\Cleaned_Json\tata-steel-limited-ir-2024"
OUTPUT_DIR = r"C:\Darsh\Techkriti_Finance\Cleaned_Json-20250327T210137Z-001\Cleaned_Json\tata-steel-limited-ir-2024_Summary" \
""

# Model selection
LLM_MODEL = "google/gemini-flash-1.5:free"

# Optional delay between API calls to potentially avoid rate limits (in seconds)
API_CALL_DELAY_SECONDS = 1

# --- Load API Key ---
API_KEY = "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"
if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in environment variables. "
        "Please create a .env file in the script's directory and add the key, "
        "or set the environment variable manually."
    )

# --- Initialize OpenAI Client for OpenRouter ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# --- Detailed Prompt Template for Summarization ---
PROMPT_TEMPLATE = """You are an expert Analyst and Content Synthesizer. Your task is to generate a high-fidelity summary of the provided text block.

Key Requirements:
1. Preserve full context and relationships between ideas
2. Include ALL significant information, key themes, goals, and figures
3. Maintain a neutral and professional tone
4. Create a coherent, logically structured summary
5. Strictly use only information from the input text

Output Guidelines:
- Provide a comprehensive summary as narrative prose
- Aim for a medium-to-large length that captures essential details
- Do not use bullet points
- Begin directly with summarized content

Input Text:
{text_to_summarize}

Summary:"""

def get_llm_summary(text_to_summarize, model_name):
    """Send text to LLM and return summary."""
    if not text_to_summarize or not text_to_summarize.strip():
        print("  Warning: Empty text provided. Returning empty summary.")
        return ""

    try:
        print(f"  Requesting summary using {model_name}")

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text_to_summarize=text_to_summarize)}]
        )

        summary = completion.choices[0].message.content.strip()
        print("  Summary received successfully.")

        if API_CALL_DELAY_SECONDS > 0:
            time.sleep(API_CALL_DELAY_SECONDS)

        return summary

    except Exception as e:
        print(f"  ERROR: API call failed. Details: {e}")
        return f"Error during summary generation: {e}"

def process_json_file(input_filepath, output_filepath, model_name):
    """Process a single JSON file for summarization."""
    print(f"Processing file: {os.path.basename(input_filepath)}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR reading file {input_filepath}: {e}")
        return False

    output_data = {}
    sections_summarized = 0
    sections_copied = 0

    for key, value in data.items():
        # Identify sections with text to summarize
        is_summarizable_section = (
            isinstance(value, list) and
            value and 
            isinstance(value[0], dict) and 
            'text' in value[0]
        )

        if is_summarizable_section:
            print(f"  Summarizing section: '{key}'")
            texts_to_join = [
                item.get('text', '') if isinstance(item, dict) else ''
                for item in value
            ]
            full_text = "\n\n".join(filter(None, texts_to_join)).strip()

            if full_text:
                summary_content = get_llm_summary(full_text, model_name)
                output_data[key] = [{"summary": summary_content}]
                sections_summarized += 1
            else:
                output_data[key] = [{"summary": ""}]
        else:
            output_data[key] = value
            sections_copied += 1

    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved summarized file: {os.path.basename(output_filepath)}")
        print(f"  Summary: Summarized={sections_summarized}, Copied={sections_copied}")
        return True

    except Exception as e:
        print(f"  ERROR saving file {output_filepath}: {e}")
        return False

def main():
    print("=" * 60)
    print("      JSON Summarization Script")
    print("=" * 60)
    print(f"Input Directory:  {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"LLM Model:        {LLM_MODEL}")
    print("-" * 60)

    # Validate and prepare directories
    if not os.path.isdir(INPUT_DIR):
        print(f"FATAL ERROR: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process files
    total_files = 0
    successful_files = 0
    failed_files = 0

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".json"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            total_files += 1
            if process_json_file(input_path, output_path, LLM_MODEL):
                successful_files += 1
            else:
                failed_files += 1

    # Final report
    print("=" * 60)
    print("Processing Complete.")
    print(f"Total JSON files:     {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed files:         {failed_files}")
    print("Summarized files saved in:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()