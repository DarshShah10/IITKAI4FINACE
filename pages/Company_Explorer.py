import streamlit as st
import os
import json
from pathlib import Path
from openai import OpenAI # Import the OpenAI library
import openai # Import for exception handling
import plotly.graph_objects as go # Import Plotly
import plotly.express as px
import pandas as pd # Useful for structuring data for some plots

# --- Configuration ---
DATA_DIR = Path("data")
METRICS_FILENAME = "metrics.json" # Define the standard metrics filename

COMPANY_FOLDERS = []
if DATA_DIR.is_dir():
    # Ensure consistent sorting regardless of OS
    COMPANY_FOLDERS = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
else:
    st.error(f"Data directory '{DATA_DIR}' not found. Please ensure it exists relative to the script.")
    st.stop()


# --- Data Loading Function (Cached) - For LLM Context ---
@st.cache_data
def load_company_data(company_name):
    """Loads data for the specified company from .txt and .json files for LLM context."""
    company_path = DATA_DIR / company_name
    combined_content = []

    if not company_path.is_dir():
        return f"Error: Directory not found for company '{company_name}' at '{company_path}'"

    combined_content.append(f"--- Start of Context for Company: {company_name} ---")

    try:
        # Ensure consistent sorting regardless of OS
        items = sorted([item for item in company_path.iterdir() if item.is_file()])
    except OSError as e:
        return f"Error listing directory contents for {company_name}: {e}"

    if not items:
        st.warning(f"No files found in the directory for company '{company_name}'.")
        combined_content.append("\n[No data files found for this company.]\n")

    for item_path in items:
        file_name = item_path.name
        file_suffix = item_path.suffix.lower()
        combined_content.append(f"\n--- Start of content from file: {file_name} ---\n")

        try:
            content_to_add = ""
            if file_suffix == ".json":
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Pretty-print JSON for better LLM readability
                        content_to_add = json.dumps(data, indent=2)
                except json.JSONDecodeError as json_e:
                    st.warning(f"Could not parse JSON file {file_name}: {json_e}. Reading as raw text.")
                    # Reset file pointer and read as raw text if JSON parsing fails
                    try:
                        with open(item_path, 'r', encoding='utf-8') as f_raw:
                             content_to_add = f_raw.read()
                    except Exception as read_raw_e:
                        st.error(f"Error re-reading JSON file {file_name} as raw text: {read_raw_e}")
                        content_to_add = f"[Error reading file after JSON parse failure: {read_raw_e}]"

                except Exception as read_e: # Catch other potential reading errors
                    st.error(f"Error processing JSON file {file_name}: {read_e}")
                    content_to_add = f"[Error processing JSON file: {read_e}]"

            elif file_suffix == ".txt":
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        content_to_add = f.read()
                except UnicodeDecodeError:
                    st.warning(f"Could not read {file_name} as UTF-8. Trying latin-1 encoding...")
                    try:
                        with open(item_path, 'r', encoding='latin-1') as f:
                            content_to_add = f.read()
                    except Exception as latin_e:
                         st.error(f"Error reading {file_name} with latin-1 encoding: {latin_e}")
                         content_to_add = f"[Error reading file with latin-1: {latin_e}]"
                except Exception as read_e:
                    st.error(f"Error reading text file {file_name}: {read_e}")
                    content_to_add = f"[Error reading text file: {read_e}]"

            else:
                st.warning(f"Skipping unsupported file type: {file_name} (only .json and .txt are supported for LLM context)")
                content_to_add = f"[Skipped unsupported file type: {file_suffix}]"

            combined_content.append(content_to_add if content_to_add else "[File was empty or unreadable]")
            combined_content.append(f"\n--- End of content from file: {file_name} ---\n")

        except Exception as e:
            # General catch-all for safety during file processing loop
            error_msg = f"Unexpected error processing file {file_name}: {e}"
            st.error(error_msg)
            combined_content.append(f"[Error processing file: {e}]")
            combined_content.append(f"\n--- End of content from file: {file_name} ---\n")

    combined_content.append(f"--- End of Context for Company: {company_name} ---")
    return "".join(combined_content)

# --- Data Loading Function (Cached) - For Metrics JSON ---
@st.cache_data
def load_metrics_data(company_name):
    """Loads and parses the metrics JSON file for the specified company."""
    metrics_path = DATA_DIR / company_name / METRICS_FILENAME
    if not metrics_path.is_file():
        # Return None if file doesn't exist, handle message in UI
        return None

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON in '{metrics_path}': {e}")
        return None # Return None on parsing error
    except Exception as e:
        st.error(f"Error reading metrics file '{metrics_path}': {e}")
        return None # Return None on other read errors


# --- Helper Function for Formatting Large Numbers ---
def format_number(num):
    """Formats large numbers into billions (B) or millions (M) for display."""
    if num is None:
        return "N/A"
    try:
        num = float(num)
        if abs(num) >= 1e9:
            return f"{num / 1e9:.2f} B"
        elif abs(num) >= 1e6:
            return f"{num / 1e6:.2f} M"
        elif abs(num) >= 1e3:
             return f"{num / 1e3:.2f} K"
        else:
            return f"{num:.2f}"
    except (ValueError, TypeError):
        return "Invalid"


# --- Plotting Functions (Unchanged from previous version) ---

def plot_income_statement_summary(income_data):
    """Generates Plotly indicators and a bar chart for Income Statement."""
    if not income_data:
        return None, None # Return None for both figure and indicators if no data

    # --- Indicators ---
    fig_indicators = go.Figure()
    net_income = income_data.get("Net Income")
    total_revenue = income_data.get("Total Revenue")
    eps = income_data.get("Basic EPS") # Or Diluted EPS

    fig_indicators.add_trace(go.Indicator(
        mode = "number",
        value = net_income,
        title = {"text": "Net Income"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, # Assuming INR currency symbol
        domain = {'row': 0, 'column': 0}))

    fig_indicators.add_trace(go.Indicator(
        mode = "number",
        value = total_revenue,
        title = {"text": "Total Revenue"},
        number = {'prefix': "₹", 'valueformat': ',.0f'},
        domain = {'row': 0, 'column': 1}))

    fig_indicators.add_trace(go.Indicator(
        mode = "number",
        value = eps,
        title = {"text": "Basic EPS"},
        number = {'prefix': "₹", 'valueformat': '.2f'},
        domain = {'row': 0, 'column': 2}))

    fig_indicators.update_layout(
        grid = {'rows': 1, 'columns': 3, 'pattern': "independent"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=150
    )

    # --- Bar Chart: Revenue vs Expenses ---
    interest_expense = income_data.get("Interest Expense", 0)
    sga_expense = income_data.get("Selling General And Administration", 0)
    tax_provision = income_data.get("Tax Provision", 0)
    # Note: Total Revenue includes Interest Income for a bank
    # Let's plot Revenue vs Major Expense Categories
    expense_categories = ['Interest Expense', 'SG&A', 'Tax Provision']
    expense_values = [interest_expense, sga_expense, tax_provision]
    revenue_value = [total_revenue, 0, 0] # Plot revenue alongside expenses for comparison

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=expense_categories,
        y=expense_values,
        name='Expenses',
        marker_color='indianred',
        text=[format_number(v) for v in expense_values],
        textposition='auto'
     ))

    fig_bar.update_layout(
        title='Major Expense Components',
        yaxis_title='Amount (INR)',
        barmode='group',
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    return fig_indicators, fig_bar

def plot_balance_sheet_summary(balance_data):
    """Generates Plotly charts for Balance Sheet."""
    if not balance_data:
        return None, None, None # Return None if no data

    # --- Indicators ---
    fig_indicators = go.Figure()
    total_assets = balance_data.get("Total Assets")
    total_liabilities = balance_data.get("Total Liabilities Net Minority Interest")
    total_equity = balance_data.get("Total Equity Gross Minority Interest")
    net_debt = balance_data.get("Net Debt")

    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = total_assets, title = {"text": "Total Assets"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 0, 'column': 0}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = total_liabilities, title = {"text": "Total Liabilities"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 0, 'column': 1}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = total_equity, title = {"text": "Total Equity"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 1, 'column': 0}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = net_debt, title = {"text": "Net Debt"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 1, 'column': 1}))

    fig_indicators.update_layout(
        grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=250
    )

    # --- Pie Chart: Assets Composition (Simplified) ---
    cash_equiv = balance_data.get("Cash Cash Equivalents And Federal Funds Sold", 0)
    investments = balance_data.get("Investments And Advances", 0) # Broad category
    net_ppe = balance_data.get("Net PPE", 0)
    goodwill_intangibles = balance_data.get("Goodwill And Other Intangible Assets", 0)
    receivables = balance_data.get("Receivables", 0)
    other_assets = total_assets - (cash_equiv + investments + net_ppe + goodwill_intangibles + receivables) if total_assets else 0

    asset_labels = ['Cash & Equiv.', 'Investments', 'Net PPE', 'Goodwill/Intangibles', 'Receivables', 'Other Assets']
    asset_values = [cash_equiv, investments, net_ppe, goodwill_intangibles, receivables, other_assets]

    fig_pie_assets = go.Figure(data=[go.Pie(labels=asset_labels,
                                             values=asset_values,
                                             hole=.3, # Donut chart
                                             pull=[0.05 if l == 'Cash & Equiv.' else 0 for l in asset_labels] # Slightly pull out cash slice
                                             )])
    fig_pie_assets.update_traces(textinfo='percent+label', hoverinfo='label+percent+value')
    fig_pie_assets.update_layout(
        title_text='Assets Composition',
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )


    # --- Bar Chart: Debt Structure ---
    total_debt = balance_data.get("Total Debt", 0)
    long_term_debt = balance_data.get("Long Term Debt", 0)
    current_debt = balance_data.get("Current Debt", 0)
    capital_lease_lt = balance_data.get("Long Term Capital Lease Obligation", 0)
    capital_lease_st = balance_data.get("Current Capital Lease Obligation", 0) # May not exist, check key
    total_lease = (capital_lease_lt if capital_lease_lt else 0) + (capital_lease_st if capital_lease_st else 0)

    debt_labels = ['Total Debt', 'Long-Term Debt', 'Current Debt', 'Capital Leases']
    debt_values = [
        total_debt if total_debt else 0,
        long_term_debt if long_term_debt else 0,
        current_debt if current_debt else 0,
        total_lease
    ]

    fig_bar_debt = go.Figure(data=[go.Bar(
        x=debt_labels,
        y=debt_values,
        marker_color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd'], # Different colors
        text=[format_number(v) for v in debt_values],
        textposition='outside' # Position text above bars
    )])
    fig_bar_debt.update_layout(
        title='Debt Structure',
        yaxis_title='Amount (INR)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )


    return fig_indicators, fig_pie_assets, fig_bar_debt


def plot_cash_flow_summary(cash_flow_data):
    """Generates Plotly charts for Cash Flow Statement."""
    if not cash_flow_data:
        return None, None # Return None if no data

    # --- Indicators ---
    fig_indicators = go.Figure()
    op_cf = cash_flow_data.get("Operating Cash Flow")
    inv_cf = cash_flow_data.get("Investing Cash Flow")
    fin_cf = cash_flow_data.get("Financing Cash Flow")
    fcf = cash_flow_data.get("Free Cash Flow")

    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = op_cf, title = {"text": "Operating CF"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 0, 'column': 0}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = inv_cf, title = {"text": "Investing CF"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 0, 'column': 1}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = fin_cf, title = {"text": "Financing CF"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 1, 'column': 0}))
    fig_indicators.add_trace(go.Indicator(
        mode = "number", value = fcf, title = {"text": "Free Cash Flow (FCF)"},
        number = {'prefix': "₹", 'valueformat': ',.0f'}, domain = {'row': 1, 'column': 1}))

    fig_indicators.update_layout(
        grid = {'rows': 2, 'columns': 2, 'pattern': "independent"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=250
    )

    # --- Bar Chart: Cash Flow Components ---
    cf_labels = ['Operating', 'Investing', 'Financing']
    cf_values = [
        op_cf if op_cf else 0,
        inv_cf if inv_cf else 0,
        fin_cf if fin_cf else 0
    ]
    colors = ['green' if v >= 0 else 'red' for v in cf_values] # Color based on positive/negative

    fig_bar_cf = go.Figure(data=[go.Bar(
        x=cf_labels,
        y=cf_values,
        marker_color=colors,
        text=[format_number(v) for v in cf_values],
        textposition='auto'
        )])

    change_in_cash = cash_flow_data.get("Changes In Cash")
    if change_in_cash:
         fig_bar_cf.add_hline(y=change_in_cash, line_dash="dot",
                              annotation_text=f"Net Change: {format_number(change_in_cash)}",
                              annotation_position="bottom right")


    fig_bar_cf.update_layout(
        title='Cash Flow from Activities',
        yaxis_title='Amount (INR)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    return fig_indicators, fig_bar_cf


# --- OpenRouter LLM Function (Unchanged) ---
def get_llm_response(context, user_query, api_key, model_name):
    """Gets a response from an OpenRouter model using the OpenAI SDK."""

    if not api_key:
        st.error("OpenRouter API key is missing. Please configure it in the sidebar.")
        return "Error: OpenRouter API key is missing."
    if not model_name:
         st.error("No LLM model selected in the sidebar.")
         return "Error: No model selected."
    if not context:
        st.warning("Cannot query LLM without company context.")
        return "Error: Company context is empty or failed to load."

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", # OpenRouter endpoint
            api_key=api_key,
        )

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful financial analyst assistant. Analyze the provided context containing financial data, summaries, news articles, and potentially other text related to a specific company. "
                    "Use ONLY the information given in the context to answer the user's query accurately and concisely. Your answer should contain no financial jargon and the language should be extremely easy so even a layman can understand. "
                    "If asked to create a story or narrative, synthesize the data points and news events from the context into a coherent timeline or explanation. "
                    "If the context doesn't contain information relevant to the query, state that the information is not available in the provided documents."
                    f"\n\n--- Start of Provided Context ---\n\n{context}\n\n--- End of Provided Context ---"
                )
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7, # Adjust for creativity vs factuality
            # max_tokens=1500, # Optional: Limit response length
        )
        return response.choices[0].message.content

    except openai.AuthenticationError:
         st.error("Authentication Error: Invalid OpenRouter API key provided. Please check your key in the sidebar or your configuration (Secrets/Environment).")
         return "Error: Invalid OpenRouter API key."
    except openai.APIConnectionError as e:
        st.error(f"API Connection Error: Could not connect to OpenRouter. Check your network and OpenRouter's status. Details: {e}")
        return f"Error: Could not connect to OpenRouter API."
    except openai.RateLimitError:
        st.error("API Rate Limit Error: You have exceeded your OpenRouter usage limit or rate limit for the selected model.")
        return "Error: API Rate limit exceeded."
    except openai.BadRequestError as e:
        st.error(f"API Bad Request Error: There might be an issue with the request format or the model input (e.g., context too long). Details: {e}")
        return f"Error: Bad request sent to API. {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred while communicating with the OpenRouter API: {e}")
        return f"Error: An unexpected error occurred: {e}"


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Company Analysis Chatbot")
st.title("📈 Company Analysis Chatbot (via OpenRouter)")
st.write("Select a company, generate a story & view metrics, or start chatting using the loaded data.")

# --- Sidebar for Configuration (Unchanged) ---
st.sidebar.header("⚙️ Configuration")
# API Key Input
api_key = ""
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    st.sidebar.success("API Key loaded from Streamlit Secrets.", icon="✅")
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        st.sidebar.success("API Key loaded from Environment Variable.", icon="✅")
    else:
        st.sidebar.warning("API Key not found in Secrets or Environment. Please enter it below.", icon="⚠️")
        api_key = st.sidebar.text_input("Enter OpenRouter API Key:", type="password", key="api_key_input", help="Get your key from https://openrouter.ai/keys")
# Model Selection
available_models = [
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3-8b-instruct",
]
default_model_name = "google/gemini-2.0-flash-exp:free"
try:
    default_index = available_models.index(default_model_name)
except ValueError:
    default_index = 0
selected_model = st.sidebar.selectbox(
    "Select LLM Model:", options=available_models, index=default_index,
    help="Choose the language model hosted on OpenRouter."
)
st.sidebar.markdown("---")
st.sidebar.info("Data is loaded from subdirectories within the 'data' folder. Financial plots use 'metrics.json'.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "show_story_and_plots" not in st.session_state:
    st.session_state.show_story_and_plots = False # Flag to control visibility
if "generated_story" not in st.session_state:
    st.session_state.generated_story = None # Store the generated story text


# --- Main Area ---

# Company Selection
if not COMPANY_FOLDERS:
    st.warning("⚠️ No company subdirectories found in the 'data' folder. Please create them and add data files (.txt, .json).")
    st.stop()

selected_company = st.selectbox(
    "Choose a Company:",
    options=COMPANY_FOLDERS,
    index=0, # Default to the first company
    key="company_select"
)

# --- Load Data (Context & Metrics) ---
# These are loaded/retrieved from cache whenever the selected_company changes or on first run
company_context = ""
metrics_data = None

if selected_company:
    # Use a spinner for combined data loading
    with st.spinner(f"⏳ Preparing data for {selected_company}..."):
        # Load text context for LLM
        company_context = load_company_data(selected_company)
        if isinstance(company_context, str) and company_context.startswith("Error:"):
             st.error(f"Failed to load text context for {selected_company}: {company_context}")
             company_context = "" # Reset context on error

        # Load structured metrics data for plotting (will be used later if button clicked)
        metrics_data = load_metrics_data(selected_company)
        # Don't display plots here yet

# --- Handle Company Change ---
# Reset state if company changes
if st.session_state.current_company != selected_company:
    st.session_state.messages = [] # Reset chat
    st.session_state.show_story_and_plots = False # Hide story/plots section
    st.session_state.generated_story = None # Clear old story
    st.session_state.current_company = selected_company
    # Display message for chat reset
    if selected_company:
         st.info(f"Switched to {selected_company}. Chat history cleared. Generate story or ask a new question.")


# --- Context Preview (Optional) ---
with st.expander("View Loaded Text Context (for LLM & Debugging)"):
    st.text_area("Context Preview", company_context if company_context else "No text context loaded or error occurred.", height=150, key="context_preview")

st.markdown("---") # Visual separator

# --- Generate Story Button ---
st.subheader(f"✍️ Generate Story & View Metrics for {selected_company}")
if st.button(f"Generate Story for {selected_company}", key="generate_story_button"):
    if not selected_company:
        st.warning("Please select a company first.")
    elif not company_context:
        st.error(f"Cannot generate story. Text context for {selected_company} failed to load or is empty.")
    elif not api_key:
         st.warning("OpenRouter API Key is missing. Please configure it in the sidebar.")
    elif not selected_model:
         st.warning("No LLM Model selected. Please configure it in the sidebar.")
    else:
        story_prompt = f"""Based *only* on the provided context documents for {selected_company}, tell the story of this company.
        # --- Start of LLM Task Definition ---
ROLE: Specialist AI: Financial Data Storyteller (Layman Audience Focus)

PRIMARY OBJECTIVE:

Analyze the provided retrieved_context_documents about {selected_company}. Synthesize this information to produce a structured, three-part output designed for maximum clarity and comprehension by an audience with no prior financial knowledge.

CORE TASK SPECIFICATIONS:

Compelling Title Generation:

Create an intriguing, attention-grabbing title that encapsulates the essence of the company's story

The title should be punchy, creative, and spark curiosity

Avoid direct financial terminology

Aim to make someone want to read more just from the title

Subheading Integration:

Add 2-3 subheadings within the main narrative section to break up the text and guide the reader

Subheadings should be:

Descriptive and engaging

Reflective of key moments or themes in the company's journey

Written in a conversational, storytelling style

Grounding: Base the entire output exclusively on the information present within the provided retrieved_context_documents. Do NOT incorporate external knowledge, assumptions, or information beyond the input text.

Output Structure: Generate the response in exactly three distinct sections, using the specified headers:

[Intriguing Title]

The Story of {selected_company}

Points to Watch

Conclusion

Section 1: "The Story of {selected_company}" - Narrative Generation: Length & Depth Specification: This section is the primary narrative and should be substantially detailed. Target an approximate length of 1000-1200 words.Content Derivation: Achieve this length by thoroughly synthesizing and elaborating upon the events, trends, challenges, successes, product details, strategic moves, and leadership actions (if mentioned) found within the retrieved_context_documents. Connect different pieces of information logically to build a cohesive journey.Narrative Style: Adopt a storytelling approach. Start with a compelling hook derived from the context. Describe the company's journey with its ups and downs. Introduce 'characters' (leadership/stakeholders) if relevant details are present in the context. Build narrative flow rather than just listing facts. Simplicity Mandate: Use simple, everyday language accessible to a complete beginner. Strictly forbid financial jargon. Explain concepts through analogy or plain description (e.g., "safety cushion" instead of PCR, "money earned from sales" instead of revenue).

Section 2: "Points to Watch" - Conceptual Considerations: Mandatory Inclusion: List exactly 2 or 3 bullet points in this section. Nature of Points: These points are NOT necessarily negative findings or problems explicitly stated in the context. They represent general business considerations or areas of vigilance relevant to the themes emerging from the story told in Section 1 (e.g., if the story highlights rapid growth, a point might be "Managing the challenges of fast expansion"; if it highlights reliance on one product, "Depending heavily on [Product Name]"). Frame them neutrally. Derivation: Derive these points conceptually from the overall narrative and context. Do NOTinvent specific data or issues. Examples: "Navigating a competitive market", "Adapting to changing technology trends mentioned", "Maintaining quality during expansion", "Reliance on key partnerships".Each point should be about 2-3 sentences long, concisely summarizing a key theme or challenge.

Section 3: "Conclusion" - Concise Summary: Length: Keep this section brief, strictly 3 to 4 sentences. Content:Summarize the main feeling or essence of the story from Section 1. Briefly reference the "Points to Watch" identified in Section 2. Ground the conclusion entirely in the preceding analysis derived from the context.

DETAILED OUTPUT SCHEMA / FORMATTING REQUIREMENTS:

# [Intriguing Title That Captures the Company's Essence]

## The Story of {selected_company}
### [First Subheading Highlighting a Key Theme]
(Narrative content)

### [Second Subheading Highlighting Another Key Moment]
(Narrative content)

### [Third Subheading Highlighting Another Key Moment]
(Narrative content)

## Points to Watch
* (Generate the first conceptual, neutrally framed point here, based on story themes.)
* (Generate the second conceptual, neutrally framed point here, based on story themes.)
* (Optional: Generate the third conceptual, neutrally framed point here, based on story themes.)

## Conclusion
(Generate the concise 3-4 sentence conclusion here, summarizing the story's essence and referencing the 'Points to Watch'.)

        """
        with st.spinner(f"✍️ Generating story for {selected_company} using {selected_model}..."):
            story = get_llm_response(company_context, story_prompt, api_key, selected_model)
            # Store the story and set the flag to show the section
            st.session_state.generated_story = story
            st.session_state.show_story_and_plots = True
            # No need to explicitly display here, the section below will handle it on rerun

# --- Story and Plots Display Section (Conditional) ---
# This section only appears if the flag is True and the company matches
if st.session_state.get('show_story_and_plots', False) and st.session_state.current_company == selected_company:

    st.markdown("---") # Separator before the combined section

    # Display the stored story
    if st.session_state.get('generated_story'):
        st.markdown("### Company Story")
        st.write(st.session_state.generated_story)
    else:
        # Should not happen if flag is True, but as a safeguard
        st.info("Story generated, but could not be displayed.")

    st.markdown("---") # Separator between story and plots
    st.subheader(f"📊 Financial Metrics Overview for {selected_company}")

    # Now, display the plots using the metrics_data loaded earlier
    if metrics_data:
        # Get data sections
        income_data = metrics_data.get("INCOME_STATEMENT")
        balance_data = metrics_data.get("BALANCE_SHEET")
        cashflow_data = metrics_data.get("CASH_FLOW")

        # --- Income Statement Plots ---
        st.markdown("#### Income Statement")
        if income_data:
            fig_income_ind, fig_income_bar = plot_income_statement_summary(income_data)
            if fig_income_ind:
                st.plotly_chart(fig_income_ind, use_container_width=True)
            if fig_income_bar:
                st.plotly_chart(fig_income_bar, use_container_width=True)
            else: st.info("Could not generate income statement charts from available data.")
        else: st.info("Income statement data not found in metrics file.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Balance Sheet Plots ---
        st.markdown("#### Balance Sheet")
        if balance_data:
            fig_bs_ind, fig_bs_pie, fig_bs_bar = plot_balance_sheet_summary(balance_data)
            col1, col2 = st.columns([1, 1])
            with col1:
                if fig_bs_ind: st.plotly_chart(fig_bs_ind, use_container_width=True)
                if fig_bs_bar: st.plotly_chart(fig_bs_bar, use_container_width=True)
            with col2:
                 if fig_bs_pie: st.plotly_chart(fig_bs_pie, use_container_width=True)
            if not fig_bs_ind and not fig_bs_pie and not fig_bs_bar:
                 st.info("Could not generate balance sheet charts from available data.")
        else: st.info("Balance sheet data not found in metrics file.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Cash Flow Plots ---
        st.markdown("#### Cash Flow Statement")
        if cashflow_data:
            fig_cf_ind, fig_cf_bar = plot_cash_flow_summary(cashflow_data)
            if fig_cf_ind: st.plotly_chart(fig_cf_ind, use_container_width=True)
            if fig_cf_bar: st.plotly_chart(fig_cf_bar, use_container_width=True)
            else: st.info("Could not generate cash flow charts from available data.")
        else: st.info("Cash flow statement data not found in metrics file.")

    else:
        # This message shows if metrics.json was missing/invalid when Generate Story was clicked
        st.warning(f"Metrics file ('{METRICS_FILENAME}') not found or could not be parsed for {selected_company}. Cannot display financial charts alongside the story.")


st.markdown("---") # Visual separator before chat

# --- Chat Interface ---
st.subheader(f"💬 Chat about {selected_company}")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input via chat input widget
if prompt := st.chat_input(f"Ask something about {selected_company}... (using {selected_model})", key="chat_input"):
    # Basic validation before processing
    if not selected_company:
        st.warning("Please select a company first.")
    elif not company_context: # Check the text context needed for the LLM
        st.warning(f"Cannot chat. Text context for {selected_company} failed to load or is empty.")
    elif not api_key:
         st.warning("OpenRouter API Key is missing. Please configure it in the sidebar to enable chat.")
    elif not selected_model:
         st.warning("No LLM Model selected. Please configure it in the sidebar to enable chat.")
    else:
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response via OpenRouter LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Use a placeholder for streaming-like effect
            message_placeholder.markdown("🧠 Thinking...")
            response = get_llm_response(company_context, prompt, api_key, selected_model)
            message_placeholder.markdown(response) # Display final response

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
