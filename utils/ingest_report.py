# ingest_report.py
import os
import re
import json
import uuid
import hashlib
import time
from tqdm import tqdm 
import logging
import sys 

# Import config and client functions/variables
import config
from config import logger # Use the logger configured in config.py

# Import specific tools
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    logger.error("langchain package not found. Please install it: pip install langchain")
    sys.exit(1)

# --- Constants ---
PINECONE_BATCH_SIZE = 64 
REPORT_BASE_DIR = r"E:\Techkriti\Cleaned_Json" 
PROMPT_DIR = r"E:\Techkriti\prompts" 
LOCAL_NARRATIVE_OUTPUT_DIR = "narrative_output" 


SECTION_FILENAMES_MAP = {
    1: ["Corporate_Overview_structured_cleaned.json"],
    2: ["Statutory_Reports_structured_cleaned.json"],
    3: ["Financial_Statements_structured_cleaned.json"],
    4: ["Other_Information_structured_cleaned.json",
        "Unclassified_structured_cleaned.json"] # Stage 4 combines these
}
# Used to determine base section name if needed, primarily for logging/sanity checks
FILENAME_SUFFIX = "_structured_cleaned.json"

# Mapping from BASE filename (used in metadata) back to the stage narrative it belongs to
CHUNK_STAGE_MAPPING = {
    "Corporate_Overview": 1,
    "Statutory_Reports": 2,
    "Financial_Statements": 3,
    "Other_Information": 4,
    "Unclassified": 4 # Chunks from Unclassified are associated with Stage 4 narrative
}

# Placeholders expected in prompt text files
PLACEHOLDER_CURRENT_TEXT = "{{CURRENT_SECTION_TEXT}}"
PLACEHOLDER_PREVIOUS_NARRATIVES = "{{PREVIOUS_NARRATIVES}}"

LOADED_PROMPTS = {} # Global dictionary to hold loaded prompts

# --- Helper Functions ---

def load_prompts(prompt_dir):
    """Loads prompt templates 1-4 from text files."""
    prompts = {}
    logger.info(f"Loading prompts from directory: {prompt_dir}")
    required_prompts = list(range(1, 5)) # Prompts 1 to 4 for this workflow
    all_loaded = True
    for i in required_prompts:
        filename = f"prompt{i}.txt"
        filepath = os.path.join(prompt_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            # Validate placeholders
            if PLACEHOLDER_CURRENT_TEXT not in content:
                logger.error(f"Prompt file {filepath} is MISSING the required placeholder '{PLACEHOLDER_CURRENT_TEXT}'.")
                all_loaded = False
            if i > 1 and PLACEHOLDER_PREVIOUS_NARRATIVES not in content:
                 logger.error(f"Prompt file {filepath} is MISSING the required placeholder '{PLACEHOLDER_PREVIOUS_NARRATIVES}'.")
                 all_loaded = False
            prompts[i] = content
            logger.info(f"Successfully loaded {filename}")
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {filepath}. Cannot proceed.")
            all_loaded = False
            # Stop early if a prompt is missing
            raise FileNotFoundError(f"Required prompt file missing: {filepath}")
        except Exception as e:
            logger.error(f"Error loading prompt file {filepath}: {e}", exc_info=True)
            all_loaded = False
            raise

    if not all_loaded: # This might be redundant if exceptions are raised above
        raise RuntimeError("One or more required prompt files (1-4) failed validation.")

    logger.info(f"Successfully loaded {len(prompts)} prompts.")
    return prompts

def generate_text_id(report_filename, main_section, original_text):
    """Generates a unique and deterministic ID for a text chunk."""
    try:
        # Use utf-8 encoding explicitly
        text_hash = hashlib.sha256(original_text.encode('utf-8', errors='replace')).hexdigest()
        # Truncate components to avoid overly long IDs (adjust lengths as needed)
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', report_filename)[:50] # Sanitize and limit filename part
        safe_section = re.sub(r'[\\/*?:"<>|]', '_', main_section)[:15] # Sanitize and limit section part
        return f"{safe_filename}_{safe_section}_{text_hash[:16]}"
    except Exception as e:
         logger.error(f"Error generating text_id: {e}", exc_info=True)
         # Fallback to UUID if hashing fails?
         return str(uuid.uuid4())

def call_llm(llm_client, formatted_prompt_content, stage_num, max_retries=3, delay=5):
    """Calls the LLM API with the fully formatted prompt content."""
    messages = [{"role": "user", "content": formatted_prompt_content}]
    prompt_len = len(formatted_prompt_content)
    logger.info(f"Calling LLM for Stage {stage_num} (model: {config.LLM_MODEL}), prompt length: {prompt_len} chars")
    if prompt_len > 32000: # Adjust threshold based on model context window / typical limits
         logger.warning(f"Prompt content for Stage {stage_num} is very long ({prompt_len} chars), may approach token limits.")

    for attempt in range(max_retries):
        try:
            start_llm_time = time.time()
            completion = llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.5, # Lower temperature for more factual narrative synthesis
                # max_tokens=4096, # Set a reasonable max_tokens limit for the response
            )
            end_llm_time = time.time()
            # Check for valid response structure before accessing content
            if not completion.choices or not completion.choices[0].message or completion.choices[0].message.content is None:
                 logger.warning(f"LLM response structure invalid or empty for Stage {stage_num} (Attempt {attempt+1}).")
                 snippet = None
            else:
                 snippet = completion.choices[0].message.content.strip()

            if not snippet:
                logger.warning(f"LLM returned empty content for Stage {stage_num} (Attempt {attempt+1}). Prompt start: {formatted_prompt_content[:150]}...")
                # Consider returning "" instead of None if DB requires non-null?
                if attempt + 1 == max_retries: return None # Return None only on final failed attempt
            else:
                 logger.info(f"LLM call for Stage {stage_num} successful (Attempt {attempt+1}). Time: {end_llm_time - start_llm_time:.2f}s. Response length: {len(snippet)} chars.")
                 return snippet # Return the valid snippet

        except openai.RateLimitError as e:
             logger.warning(f"LLM Rate Limit Error (Stage {stage_num}, Attempt {attempt+1}/{max_retries}): {e}. Retrying after delay...")
             # Implement longer backoff for rate limits
             current_delay = delay * (2 ** attempt) # Exponential backoff
             logger.warning(f"Waiting {current_delay}s before retry...")
             time.sleep(current_delay)
        except openai.APIError as e:
            logger.error(f"LLM API Error (Stage {stage_num}, Attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
            # Decide if API errors are retryable, maybe retry once?
            if attempt + 1 == max_retries: return None
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            # Catch other potential exceptions (network issues, etc.)
            logger.error(f"Unexpected Error during LLM call (Stage {stage_num}, Attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
            if attempt + 1 == max_retries: return None # Fail after retries
            time.sleep(delay * (attempt + 1))

    logger.error(f"LLM call failed for Stage {stage_num} after {max_retries} attempts.")
    return None # Indicate failure

def store_texts_in_db(cursor, text_id, report_filename, company_name, report_year, original_text, associated_full_narrative):
    """Stores original chunk and the full narrative for its stage in the DB."""
    logger.debug(f"Storing text_id {text_id} in DB...")
    try:
        # Ensure narrative is not None, use empty string if it is
        narrative_to_store = associated_full_narrative if associated_full_narrative is not None else ""
        cursor.execute("""
            INSERT INTO report_chunks (text_id, report_filename, company_name, report_year, original_text, intermediate_story_snippet)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (text_id) DO UPDATE SET
                report_filename = EXCLUDED.report_filename,
                company_name = EXCLUDED.company_name,
                report_year = EXCLUDED.report_year,
                original_text = EXCLUDED.original_text,
                intermediate_story_snippet = EXCLUDED.intermediate_story_snippet;
        """, (text_id, report_filename, company_name, report_year, original_text, narrative_to_store))
        logger.debug(f"DB upsert prepared for {text_id}.")
    except psycopg2.Error as e:
        logger.error(f"DB Upsert failed for text_id {text_id}: {e.pgcode} - {e.pgerror}", exc_info=True)
        # Rethrowing allows the main loop to handle rollback for the batch
        raise ConnectionError(f"Database upsert error for {text_id}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during DB store for {text_id}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error storing {text_id} in DB: {e}") from e


def upsert_batch_to_pinecone(index_obj, pinecone_batch, db_conn, namespace, max_retries=3, delay=5):
    """Upserts a batch to Pinecone index object and commits DB on success."""
    if not pinecone_batch:
        logger.debug("Upsert batch is empty, skipping.")
        return 0

    batch_size = len(pinecone_batch)
    logger.info(f"Attempting Pinecone upsert: Batch Size={batch_size}, Namespace='{namespace}'...")

    for attempt in range(max_retries):
        try:
            start_upsert_time = time.time()
            # Upsert using the passed index object
            upsert_response = index_obj.upsert(vectors=pinecone_batch, namespace=namespace)
            end_upsert_time = time.time()
            logger.info(f"Pinecone upsert successful (Attempt {attempt+1}): Response={upsert_response}. Time={end_upsert_time - start_upsert_time:.2f}s")

            # --- Commit DB changes AFTER successful Pinecone upsert ---
            try:
                 logger.info("Committing database changes for the processed batch...")
                 db_conn.commit()
                 logger.info("Database commit successful.")
            except psycopg2.Error as db_e:
                 # This is critical - data is in Pinecone but DB commit failed.
                 logger.critical(f"FATAL: DB commit failed after successful Pinecone upsert: {db_e.pgcode} - {db_e.pgerror}", exc_info=True)
                 # Requires manual intervention or more complex reconciliation.
                 raise ConnectionError("DB commit failed post-Pinecone success. Data inconsistency likely.") from db_e # Propagate critical error

            return batch_size # Return number of vectors upserted successfully

        except ApiException as e:
            # Handle Pinecone specific API errors
            logger.warning(f"Pinecone API Error during upsert (Attempt {attempt+1}/{max_retries}): Status={e.status}, Reason={e.reason}, Body={e.body}")
            # Decide on retry logic based on status code if needed (e.g., don't retry on 4xx?)
            if e.status >= 400 and e.status < 500:
                 logger.error("Pinecone returned client-side error (4xx). Not retrying.")
                 break # Break retry loop for client errors
            # Continue retry for server errors (5xx) or timeouts
        except Exception as e:
            logger.warning(f"Unexpected error during Pinecone upsert (Attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
            # Continue retrying for generic errors

        # If loop continues (retry needed)
        if attempt + 1 < max_retries:
             current_delay = delay * (2 ** attempt) # Exponential backoff
             logger.info(f"Waiting {current_delay}s before Pinecone upsert retry...")
             time.sleep(current_delay)

    # If loop finishes without success
    logger.error(f"Pinecone upsert failed after {max_retries} attempts for namespace '{namespace}'. Rolling back DB.")
    try:
        db_conn.rollback()
        logger.warning("Database rollback successful due to Pinecone failure.")
    except psycopg2.Error as db_e:
        logger.critical(f"FATAL: DB rollback failed after Pinecone upsert failure: {db_e.pgcode} - {db_e.pgerror}", exc_info=True)
        # This indicates a potentially inconsistent state. Manual check needed.
    return 0 # Indicate failure

def read_text_from_section(report_dir, stage_num):
    """Reads and concatenates all text from JSON files mapped to a stage."""
    all_text_content = []
    json_filenames = SECTION_FILENAMES_MAP.get(stage_num, [])
    if not json_filenames:
        logger.warning(f"No JSON files mapped for stage {stage_num}.")
        return "" # Return empty string if no files are mapped

    logger.info(f"Reading text for Stage {stage_num} from: {json_filenames}")
    for json_filename in json_filenames:
        file_path = os.path.join(report_dir, json_filename)
        if not os.path.exists(file_path):
            # Make this a more prominent warning if the file *is* expected
            logger.warning(f"JSON file not found for Stage {stage_num}: {file_path}. Skipping this file.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract text robustly
            count = 0
            for key, value in json_data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'text' in item and item['text']:
                            all_text_content.append(item['text'].strip())
                            count += 1
            logger.info(f"Read {count} text chunks from {json_filename}.")

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error in {file_path} for Stage {stage_num}: {e}. Skipping file.")
        except Exception as e:
            logger.error(f"Failed to read/process {file_path} for Stage {stage_num}: {e}", exc_info=True)
            # Decide if this should halt processing for the stage
            # For now, we continue with potentially partial text

    # Join extracted text with separators
    full_text = "\n\n---\n\n".join(all_text_content) # Use a clear separator
    logger.info(f"Total combined text length for Stage {stage_num}: {len(full_text)} chars.")
    return full_text

# --- Main Processing Function (Revised Workflow) ---

def process_report(report_dir, llm_client, embedding_model, pinecone_index_obj, db_conn, prompts):
    """
    Processes a single report directory:
    A. Reads all text for Stages 1-4.
    B. Generates sequential narratives (Acts 1-4) using LLM, saving locally.
    C. Processes original chunks for DB storage (original text + full stage narrative)
       and Pinecone upsert (original text embedding + metadata).
    D. Handles final batch upsert.
    """
    report_filename = os.path.basename(report_dir)
    logger.info(f"--- Starting processing: Report='{report_filename}' ---")
    processing_start_time = time.time()

    # --- Basic Setup & Validation ---
    match = re.search(r'_(.+?)_(\d{4})_(\d{4})_', report_filename)
    if not match:
        logger.error(f"Could not parse Company/Year from report filename: '{report_filename}'. Skipping report.")
        return 0 # Return 0 vectors processed
    company_name = match.group(1)
    report_year = f"{match.group(2)}-{match.group(3)}"
    # Create a safe namespace name for Pinecone
    pinecone_namespace = re.sub(r'[^a-zA-Z0-9_-]+', '_', company_name).lower() # Lowercase recommended
    logger.info(f"Identified - Company: {company_name}, Year: {report_year}, Pinecone Namespace: {pinecone_namespace}")

    # Create local output directory for narratives
    local_save_path = os.path.join(report_dir, LOCAL_NARRATIVE_OUTPUT_DIR)
    try:
        os.makedirs(local_save_path, exist_ok=True)
        logger.info(f"Local narrative output directory: {local_save_path}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{local_save_path}': {e}. Cannot save narratives. Skipping report.")
        return 0

    # --- Phase A: Read all text for Stages 1-4 ---
    logger.info("--- Phase A: Reading text for stages 1-4 ---")
    phase_a_start = time.time()
    all_texts = {}
    reading_successful = True
    for stage_num in range(1, 5):
        try:
            all_texts[stage_num] = read_text_from_section(report_dir, stage_num)
            if not all_texts[stage_num] and stage_num < 4: # Be more concerned if early stages are empty
                 logger.warning(f"No text content found for critical Stage {stage_num}. Narrative quality may suffer.")
                 # Consider if this should be fatal: reading_successful = False; break
        except Exception as e:
             logger.error(f"Error during text reading phase for Stage {stage_num}: {e}. Aborting report.", exc_info=True)
             reading_successful = False
             break
    phase_a_duration = time.time() - phase_a_start
    logger.info(f"--- Phase A Complete ({phase_a_duration:.2f}s) ---")
    if not reading_successful: return 0

    # --- Phase B: Generate Narratives Sequentially (Acts 1-4) ---
    logger.info("--- Phase B: Generating sequential narratives (Acts 1-4) ---")
    phase_b_start = time.time()
    narratives = {1: None, 2: None, 3: None, 4: None} # Store generated full narratives
    cumulative_previous_narratives = ""
    narrative_gen_failed = False

    for stage_num in range(1, 5):
        stage_start_time = time.time()
        logger.info(f"--- Generating Narrative for Stage {stage_num} ---")
        prompt_template = prompts.get(stage_num)
        current_stage_text = all_texts.get(stage_num, "")

        if not prompt_template:
            logger.error(f"Missing prompt template for stage {stage_num}. Skipping narrative.")
            continue # Skip this stage's narrative

        if not current_stage_text:
             # Allow generating narrative even if text is empty? Depends on prompt.
             # For now, skip if no text.
             logger.warning(f"No input text for Stage {stage_num}. Skipping narrative generation.")
             continue

        # Format the prompt robustly
        formatted_prompt = prompt_template
        try:
            if PLACEHOLDER_CURRENT_TEXT in formatted_prompt:
                 formatted_prompt = formatted_prompt.replace(PLACEHOLDER_CURRENT_TEXT, current_stage_text, 1)
            else: logger.warning(f"Prompt {stage_num} missing '{PLACEHOLDER_CURRENT_TEXT}'.")

            if stage_num > 1:
                 if PLACEHOLDER_PREVIOUS_NARRATIVES in formatted_prompt:
                     # Provide context clearly
                     prev_narr_context = cumulative_previous_narratives if cumulative_previous_narratives else "No previous narratives generated."
                     formatted_prompt = formatted_prompt.replace(PLACEHOLDER_PREVIOUS_NARRATIVES, prev_narr_context, 1)
                 else: logger.warning(f"Prompt {stage_num} missing '{PLACEHOLDER_PREVIOUS_NARRATIVES}'.")

            # Call LLM
            stage_narrative = call_llm(llm_client, formatted_prompt, stage_num)

            if stage_narrative:
                narratives[stage_num] = stage_narrative
                logger.info(f"Successfully generated narrative for Stage {stage_num}.")
                # Save narrative locally
                save_filename = os.path.join(local_save_path, f"act_{stage_num}_narrative.txt")
                try:
                    with open(save_filename, 'w', encoding='utf-8') as f:
                        f.write(stage_narrative)
                    logger.info(f"Saved Stage {stage_num} narrative locally: '{save_filename}'")
                except Exception as e:
                    logger.error(f"Failed to save Stage {stage_num} narrative locally: {e}")

                # Append to cumulative narratives with clear separation
                cumulative_previous_narratives += f"\n\n=== ACT {stage_num} NARRATIVE ===\n{stage_narrative}\n=== END ACT {stage_num} ==="
            else:
                logger.error(f"Failed to generate narrative for Stage {stage_num}. Subsequent stages may lack context.")
                # Mark failure? Allow continuation?
                narrative_gen_failed = True # Mark that at least one failed

        except Exception as e:
             logger.error(f"Unexpected error during narrative generation for Stage {stage_num}: {e}", exc_info=True)
             narrative_gen_failed = True
        logger.info(f"--- Stage {stage_num} Narrative Generation took {time.time() - stage_start_time:.2f}s ---")

    phase_b_duration = time.time() - phase_b_start
    logger.info(f"--- Phase B Complete ({phase_b_duration:.2f}s). Narrative Gen Failures: {narrative_gen_failed} ---")
    # Optionally stop if narrative generation failed critically: if narrative_gen_failed: return 0

    # --- Phase C: Process Chunks for Upsert (Embedding + DB + Pinecone) ---
    logger.info("--- Phase C: Processing Chunks for Upsert ---")
    phase_c_start = time.time()
    pinecone_batch = []
    total_vectors_processed = 0
    processed_chunk_count = 0
    db_cursor = db_conn.cursor() # Get cursor for batch operations

    try:
        # Iterate through the stages/files again to get individual chunks
        # Use tqdm for chunk processing progress if many chunks expected
        chunk_iterator = []
        for stage_num_for_chunk, json_filenames in SECTION_FILENAMES_MAP.items():
             for json_filename in json_filenames:
                  chunk_iterator.append({'stage': stage_num_for_chunk, 'filename': json_filename})

        # Process chunks file by file
        for file_info in tqdm(chunk_iterator, desc=f"Processing Chunks in {report_filename}"):
            stage_num_for_chunk = file_info['stage']
            json_filename = file_info['filename']

            # Retrieve the full narrative generated earlier for this stage
            associated_full_narrative = narratives.get(stage_num_for_chunk)
            # Decide handling if narrative is missing: skip chunks? associate empty?
            if associated_full_narrative is None:
                logger.warning(f"Skipping chunks in {json_filename} as narrative for Stage {stage_num_for_chunk} is missing.")
                continue # Skip all chunks in this file

            file_path = os.path.join(report_dir, json_filename)
            if not os.path.exists(file_path): continue # Should have been caught earlier, but double check

            # Determine base section name for metadata
            base_section_name = json_filename.replace(FILENAME_SUFFIX, "")
            chunk_processing_stage = CHUNK_STAGE_MAPPING.get(base_section_name)
            if chunk_processing_stage is None:
                 logger.warning(f"Chunk stage mapping error for {base_section_name}. Skipping {json_filename}.")
                 continue
            # Sanity check: Ensure the mapping aligns with the narrative we have
            if chunk_processing_stage != stage_num_for_chunk:
                 logger.error(f"Logic Error: Chunk mapping stage {chunk_processing_stage} != Narrative stage {stage_num_for_chunk} for file {json_filename}. Skipping.")
                 continue

            logger.debug(f"Processing chunks from {json_filename} (Stage {chunk_processing_stage})")
            try:
                with open(file_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to reload {file_path} for chunk processing: {e}. Skipping.")
                continue

            # Iterate through sub-sections and chunks within the file
            for sub_section_name, chunks_list in json_data.items():
                if not isinstance(chunks_list, list): continue

                for chunk_index, chunk_data in enumerate(chunks_list):
                    if not isinstance(chunk_data, dict) or 'text' not in chunk_data: continue
                    original_text = chunk_data['text'].strip()
                    if not original_text: continue # Skip empty chunks

                    processed_chunk_count += 1
                    source_type = "Annual Report" # Or "Press Release" if PR logic added back

                    # Generate IDs
                    text_id = generate_text_id(report_filename, base_section_name, original_text)
                    vector_id = f"vec_{text_id}" # Pinecone vector ID

                    # Store Texts in DB (Original Chunk + FULL Stage Narrative)
                    # Errors raised here will trigger batch rollback below
                    store_texts_in_db(db_cursor, text_id, report_filename, company_name, report_year,
                                      original_text, associated_full_narrative)

                    # Embed Raw Text Chunk
                    try:
                        embedding_vector = embedding_model.encode(original_text).tolist()
                    except Exception as e:
                        logger.error(f"Embedding failed for chunk {text_id}: {e}. Skipping vector.", exc_info=True)
                        # Rollback the single DB insert attempt for this chunk
                        try: db_conn.rollback()
                        except Exception as rb_e: logger.error(f"Rollback failed after embedding error: {rb_e}")
                        continue # Skip to next chunk

                    # Prepare Pinecone Metadata
                    metadata = {
                        "report_filename": report_filename, "company_name": company_name,
                        "report_year": report_year, "source_type": source_type,
                        "main_section": base_section_name, # Section chunk came from
                        "sub_section": sub_section_name, # Subsection (JSON key)
                        "intermediate_story_stage": chunk_processing_stage, # Stage of the associated narrative
                        "text_id": text_id, # Link back to DB table
                        # Flags for potential filtering later
                        "is_factual_basis": base_section_name in ["Financial_Statements", "Statutory_Reports"],
                        "is_narrative": base_section_name in ["Corporate_Overview", "Press Release"], # Add PR if used
                        "is_other_info": base_section_name in ["Other_Information", "Unclassified"],
                    }

                    # Add to Pinecone Batch
                    pinecone_batch.append((vector_id, embedding_vector, metadata))

                    # Check Batch Size & Upsert/Commit
                    if len(pinecone_batch) >= PINECONE_BATCH_SIZE:
                        upserted_count = upsert_batch_to_pinecone(pinecone_index_obj, pinecone_batch, db_conn, pinecone_namespace)
                        if upserted_count > 0:
                            total_vectors_processed += upserted_count
                            pinecone_batch.clear() # Clear batch on successful upsert+commit
                        else:
                            # upsert_batch_to_pinecone handles logging and rollback
                            logger.error(f"Stopping report '{report_filename}' due to persistent Pinecone upsert failure.")
                            raise RuntimeError("Pinecone upsert failed repeatedly.") # Trigger outer exception handling

        logger.info(f"Processed {processed_chunk_count} text chunks for embedding/upsert.")

    except (RuntimeError, ConnectionError) as e: # Catch specific errors from helpers
        # Errors related to DB or Pinecone during batch processing
        logger.error(f"Report processing failed during Phase C: {e}", exc_info=True)
        # No need for explicit rollback here, upsert_batch handles it or outer finally does
        raise # Reraise to stop processing this report and trigger outer finally
    except Exception as e:
         logger.error(f"Unexpected error during chunk processing (Phase C): {e}", exc_info=True)
         try: db_conn.rollback() # Attempt rollback for any partial batch inserts
         except Exception as rb_e: logger.error(f"Rollback failed after Phase C error: {rb_e}")
         raise # Reraise

    finally:
        # Ensure cursor is closed after Phase C, regardless of success/failure within C
        if db_cursor:
             try: db_cursor.close()
             except Exception as cur_e: logger.error(f"Error closing DB cursor: {cur_e}")
        phase_c_duration = time.time() - phase_c_start
        logger.info(f"--- Phase C Complete ({phase_c_duration:.2f}s) ---")


    # --- Phase D: Final Batch Upsert ---
    # This runs only if Phase C completed without raising an exception
    logger.info("--- Phase D: Upserting final batch ---")
    phase_d_start = time.time()
    try:
        if pinecone_batch:
            upserted_count = upsert_batch_to_pinecone(pinecone_index_obj, pinecone_batch, db_conn, pinecone_namespace)
            if upserted_count > 0:
                total_vectors_processed += upserted_count
                pinecone_batch.clear()
            else:
                 # Log error, but don't necessarily stop pipeline for final batch failure
                 logger.error(f"Final Pinecone batch upsert failed for report '{report_filename}'.")
        else:
             logger.info("Final batch is empty, nothing to upsert.")
    except Exception as e:
        logger.error(f"Error during final upsert (Phase D): {e}", exc_info=True)
        # Consider if rollback is needed here? Depends if DB interactions happened.
        # If upsert_batch_to_pinecone handles rollback on fail, maybe not needed here.

    phase_d_duration = time.time() - phase_d_start
    processing_duration = time.time() - processing_start_time
    logger.info(f"--- Phase D Complete ({phase_d_duration:.2f}s) ---")
    logger.info(f"--- Finished Report: '{report_filename}'. Vectors Upserted: {total_vectors_processed}. Total Time: {processing_duration:.2f}s ---")

    return total_vectors_processed


# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()
    logger.info("====== Financial Report Ingestion Pipeline Starting ======")
    grand_total_vectors = 0
    processed_dirs_count = 0
    report_dirs = [] # Initialize

    try:
        # 0. Load Prompts (1-4)
        if not os.path.isdir(PROMPT_DIR):
             logger.critical(f"Prompt directory not found: '{PROMPT_DIR}'. Exiting.")
             sys.exit(1)
        LOADED_PROMPTS = load_prompts(PROMPT_DIR) # Will raise error if loading fails

        # 1. Test Setup
        logger.info("====== Running Initial Configuration Test ======")
        if not config.test_setup():
             logger.critical("Initial configuration test failed. Cannot continue. Exiting.")
             sys.exit(1)
        logger.info("====== Configuration Test Passed ======")

        # 2. Get Clients (after successful test)
        logger.info("====== Initializing Clients ======")
        llm_client = config.get_llm_client()
        embedding_model = config.get_embedding_model()
        pinecone_index_obj = config.get_pinecone_index() # Get the specific index object
        db_conn = config.get_db_connection()
        logger.info("====== Clients Initialized Successfully ======")

        # 3. Find Report Directories
        if not os.path.isdir(REPORT_BASE_DIR):
            logger.critical(f"Report base directory not found: '{REPORT_BASE_DIR}'. Exiting.")
            sys.exit(1)
        # List only directories
        report_dirs = [d for d in os.listdir(REPORT_BASE_DIR) if os.path.isdir(os.path.join(REPORT_BASE_DIR, d))]
        logger.info(f"Found {len(report_dirs)} potential report directories in '{REPORT_BASE_DIR}'.")
        if not report_dirs:
            logger.warning("No report directories found to process. Exiting.")
            sys.exit(0)

        # 4. Process Each Report
        logger.info("====== Starting Report Processing Loop ======")
        # Use tqdm for the outer loop over report directory names
        for dir_name in tqdm(report_dirs, desc="Processing Reports", unit="report"):
            report_dir_path = os.path.join(REPORT_BASE_DIR, dir_name)
            try:
                report_vectors = process_report(
                    report_dir_path,
                    llm_client,
                    embedding_model,
                    pinecone_index_obj, # Pass the index object
                    db_conn,
                    LOADED_PROMPTS
                )
                grand_total_vectors += report_vectors
                processed_dirs_count += 1
            except KeyboardInterrupt:
                 # Allow graceful exit on Ctrl+C during report processing
                 logger.warning(f"KeyboardInterrupt received during processing of '{dir_name}'. Stopping loop.")
                 break # Exit the report processing loop
            except Exception as report_e:
                 # Catch errors from process_report to allow pipeline to continue (optional)
                 logger.error(f"Failed to process report '{dir_name}' due to error: {report_e}. Skipping to next report.", exc_info=True)
                 # Ensure DB state is clean before next iteration (rollback might have occurred in process_report)
                 try:
                     db_conn.rollback() # Attempt rollback just in case
                 except Exception: pass # Ignore rollback errors here

        logger.info("====== Report Processing Loop Finished ======")

    except (RuntimeError, FileNotFoundError, ConnectionError, KeyboardInterrupt) as e:
        logger.critical(f"Pipeline execution stopped due to critical error or interruption: {e}", exc_info=True)
        if isinstance(e, KeyboardInterrupt):
             logger.warning("Pipeline interrupted by user (Ctrl+C).")
        # Exit code can indicate error
        exit_code = 1
    except Exception as e:
         logger.critical(f"An unexpected critical error occurred in the main block: {e}", exc_info=True)
         exit_code = 1
    else:
         # No critical errors in the main try block
         exit_code = 0
    finally:
        # Ensure DB connection is always closed
        logger.info("Performing final cleanup: Closing database connection...")
        config.close_db_connection()

        # Final Summary
        main_end_time = time.time()
        duration = main_end_time - main_start_time
        logger.info(f"====== Pipeline Finished ======")
        logger.info(f"Successfully processed {processed_dirs_count} / {len(report_dirs)} directories.")
        logger.info(f"Total vectors upserted across all reports: {grand_total_vectors}")
        logger.info(f"Total pipeline execution time: {duration:.2f} seconds")
        logger.info(f"====== Exiting Script (Code: {exit_code}) ======")
        sys.exit(exit_code) # Exit with 0 on success, 1 on error