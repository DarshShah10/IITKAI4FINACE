import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load variables from .env file

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_MODEL = "google/gemini-2.0-flash-001"  # Or "google/gemini-pro"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

# --- Initialize OpenRouter Client ---
# The OpenAI library is compatible with OpenRouter's API structure
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={ # Optional, but good practice for OpenRouter
        # "HTTP-Referer": OPENROUTER_REFERRER,
        # "X-Title": "NSE Symbol Identifier", # Optional project name
    },
)

# --- Define the Tool (Function) the AI can Call ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "identify_nse_symbol",
            "description": (
                "Identifies and returns the NSE stock symbol from a provided list "
                "that corresponds to the company discussed in the annual report text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": (
                            "The NSE stock symbol (e.g., 'RELIANCE', 'INFY') identified "
                            "as belonging to the company in the annual report."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "The confidence level of the match (high, medium, or low)."
                    }
                },
                "required": ["symbol", "confidence"],
            },
        },
    }
]

# --- Define the Agent Function ---
def find_company_symbol_from_report(report_text: str, nse_symbols: list[str]) -> tuple[str | None, str | None]:
    """
    Uses an AI model via OpenRouter to find the matching NSE symbol for an annual report.

    Args:
        report_text: The textual content of the annual report (or a relevant snippet).
        nse_symbols: A list of potential NSE symbols to choose from.

    Returns:
        A tuple containing:
        - The identified NSE symbol (str) or None if not found/confident.
        - The confidence level (str) or None.
    """
    symbol_list_str = ", ".join(nse_symbols)

    prompt = f"""
    Analyze the following annual report text snippet and determine which of the provided NSE symbols
    most likely corresponds to the company being discussed.

    Consider the company name, subsidiaries, brand names, and any other identifiers present in the text.
    Match these identifiers against the provided list of NSE symbols. Choose the single best match.

    Annual Report Text Snippet:
    ---
    {report_text[:4000]} 
    ---
    (Note: Text might be truncated for brevity in this prompt)

    List of Potential NSE Symbols:
    [{symbol_list_str}]

    You MUST use the 'identify_nse_symbol' tool to return your answer, indicating the matched symbol
    and your confidence level (high, medium, low) based on the evidence in the text. If you are
    uncertain or cannot find a clear match, use 'low' confidence.
    """
    # Truncate report_text if too long for the model's context window,
    # focusing on the beginning which often has key identifiers.
    # Adjust the 4000 character limit as needed based on model and typical report structure.


    try:
        print("--- Sending request to OpenRouter ---")
        response = client.chat.completions.create(
            model=GOOGLE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst AI specializing in Indian NSE listed companies. Your task is to accurately identify the company symbol associated with an annual report using the provided tools."},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "identify_nse_symbol"}} # Force tool use
        )

        print("--- Received response from OpenRouter ---")
        # print(response) # Uncomment for debugging

        message = response.choices[0].message

        # Check if the model decided to call the tool
        if message.tool_calls:
            tool_call = message.tool_calls[0] # Assuming only one tool call is expected
            if tool_call.function.name == "identify_nse_symbol":
                function_args = json.loads(tool_call.function.arguments)
                identified_symbol = function_args.get("symbol")
                confidence = function_args.get("confidence")

                # Validate if the returned symbol is actually in the original list
                if identified_symbol and identified_symbol.upper() in [s.upper() for s in nse_symbols]:
                     print(f"--- Tool call successful: Symbol={identified_symbol}, Confidence={confidence} ---")
                     return identified_symbol, confidence
                elif identified_symbol:
                     print(f"--- Warning: Model returned symbol '{identified_symbol}' not in the provided list. ---")
                     # Decide how to handle this - return None or the potentially incorrect symbol?
                     # Let's return None for stricter matching.
                     return None, "low" # Indicate low confidence due to mismatch
                else:
                     print("--- Tool call arguments missing 'symbol'. ---")
                     return None, None

        print("--- Model did not call the expected tool. ---")
        # print("Model Response:", message.content) # See what the model said instead
        return None, None

    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Reliance Industries
    report_snippet_1 = '''"section_name": "Corporate Overview",
    "filename": "AR_24043_ADANIGREEN_2023_2024_28052024191142",
    "extraction_method": "Keyword/Fuzzy",
    "content": [
        {
            "page_number": 9,
            "identified_as": "Report Highlights / Year at a Glance",
            "text": "Delivering Stronger Than Ever Performance \nConsistent High \nOperational \nPerformance \nCUF: Capacity Utilisation Factor \n10,934 MW\nOperational capacity \nbecame the first \ncompany in India to \ncross the 10,000 MW of \noperational renewable \nenergy capacity\n35%\n2,848 MW\nHighest-ever renewable \ncapacity addition \n21,806 \nmillion units \nSale of energy \n47%\n24.5%\nSolar portfolio CUF\n20 basis points\n29.4%\nWind portfolio CUF\n420 basis points\n40.7%\nHybrid (solar-wind) \nportfolio CUF\n520 basis points\n2nd largest \nSolar PV \ndeveloper \nGlobal ranking in  \nMercom Capital Group’s \nlatest Global Annual \nReport 2022-23\n06\nHIGHLIGHTS OF THE YEAR\n"
'''

    import pandas as pd
    nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    # Read the CSV file directly from NSE India
    df = pd.read_csv(nse_url)

    # Extract stock symbols (Ticker column is usually 'SYMBOL')
    nse_tickers = df['SYMBOL'].tolist()

    print("\n--- Example 1: Reliance ---")
    symbol, confidence = find_company_symbol_from_report(report_snippet_1, nse_tickers)
    if symbol:
        print(f"Identified Symbol: {symbol} (Confidence: {confidence})")
    else:
        print("Could not identify the symbol.")