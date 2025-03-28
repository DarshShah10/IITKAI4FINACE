# pages/3_🎓_Learn_Finance.py
import streamlit as st
from openai import OpenAI
import re # Import regex for key generation

# --- Helper Functions ---

def get_api_key():
    """Retrieves the API key securely."""
    # Use st.secrets for secure key management
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OpenRouter API key not found. Please add it to your Streamlit secrets (.streamlit/secrets.toml). Example: OPENROUTER_API_KEY='sk-or-...'")
        st.stop()
    return api_key

# Function to generate a safe key from topic text
def generate_safe_key(text):
    # Remove non-alphanumeric characters and replace spaces with underscores
    return re.sub(r'\W+', '', text.lower().replace(' ', '_'))

@st.cache_data(show_spinner=False) # Cache explanations, hide default spinner
def explain_concept(concept, api_key):
    """Uses LLM to explain a financial concept simply."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        messages = [
            {
                "role": "system",
                "content": """You are an excellent teacher explaining financial concepts to a complete beginner, perhaps a 10-12 year old.
                Use very simple language, short sentences, and relatable analogies (like lemonade stands, piggy banks, allowances).
                Avoid jargon. Focus on the core idea. Be friendly and encouraging.
                Format the explanation clearly, using markdown (like bullet points * Key takeaway) for key takeaways.
                Keep the explanation concise, aiming for under 300 words."""
            },
            {
                "role": "user",
                "content": f"Explain this financial concept in simple terms: {concept}"
            }
        ]
        response = client.chat.completions.create(
            model="google/gemini-flash-1.5", # Fast model
            messages=messages,
            temperature=0.6,
            max_tokens=400, # Increased slightly for potential formatting
            stream=False # Get the full response at once
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Sorry, failed to get explanation for '{concept}'. Error: {e}", icon="🚨")
        return None # Return None on error

# No caching for follow-up as context changes
def ask_follow_up(concept, initial_explanation, follow_up_question, api_key):
    """Uses LLM to answer a follow-up question based on the initial explanation."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant. You previously provided an explanation of a financial concept in simple terms.
                Now, the user has a follow-up question about that explanation.
                Answer the user's question clearly and simply, maintaining the same easy-to-understand tone (like explaining to a 10-12 year old).
                Refer back to the initial explanation context if needed. Keep the answer focused and concise."""
            },
            {
                "role": "assistant", # Provide the initial explanation as context
                "content": f"Okay, I explained '{concept}' like this:\n\n{initial_explanation}"
            },
            {
                "role": "user", # The user's follow-up question
                "content": follow_up_question
            }
        ]
        response = client.chat.completions.create(
            model="google/gemini-flash-1.5",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Sorry, failed to answer follow-up. Error: {e}", icon="🚨")
        return None # Return None on error

# --- Streamlit Page UI ---
st.set_page_config(layout="wide", page_title="Learn Finance", page_icon="🎓")

st.title("🎓 Learn Finance Basics")
st.markdown("Confused by financial terms? Choose a topic below, get a simple explanation, and ask follow-up questions!")
st.divider()

# Get API Key once
API_KEY = get_api_key()

# List of common financial questions/topics
topics = [
    "What is Revenue?",
    "What is Profit (Net Income)?",
    "What are Assets?",
    "What are Liabilities?",
    "What is Equity?",
    "How to Read a Balance Sheet?",
    "What is an Income Statement?",
    "What is Cash Flow?",
    "What is a Stock?",
    "What is a P/E Ratio (Price-to-Earnings)?",
    "What are Dividends?",
    "What are Bonds?",
    "Red Flags in Financial Statements!",
    "What is Inflation?",
    "What is Interest?",
]

# Initialize session state for storing explanations and chat history per topic
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {} # Stores list of {"role": "user/assistant", "content": "..."}

# --- UI Layout ---
selected_topic = st.selectbox("Choose a financial topic to learn about:", topics)

if selected_topic:
    topic_key = generate_safe_key(selected_topic) # Generate safe key for session state

    # Button to get initial explanation (outside expander for better visibility)
    if st.button(f"Explain '{selected_topic}'", key=f"explain_{topic_key}", type="primary"):
        with st.spinner(f"Asking the expert to explain '{selected_topic}'..."):
            explanation = explain_concept(selected_topic, API_KEY)
            if explanation:
                st.session_state.explanations[topic_key] = explanation
                # Reset chat history when getting a new explanation
                st.session_state.chat_histories[topic_key] = []
            else:
                # Clear potentially stale explanation if API call failed
                if topic_key in st.session_state.explanations:
                    del st.session_state.explanations[topic_key]
                if topic_key in st.session_state.chat_histories:
                    del st.session_state.chat_histories[topic_key]

    st.divider()

    # Display explanation and chat area if an explanation exists for the topic
    if topic_key in st.session_state.explanations:
        current_explanation = st.session_state.explanations[topic_key]

        # Container for the explanation and chat
        with st.container(border=True):
            st.subheader(f"Understanding: {selected_topic}")
            st.info(current_explanation, icon="💡") # Use st.info for visual distinction

            st.markdown("*Got questions about this explanation? Ask below!*")

            # Display chat history
            if topic_key in st.session_state.chat_histories:
                for message in st.session_state.chat_histories[topic_key]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Chat input for follow-up questions
            prompt = st.chat_input(f"Ask a question about '{selected_topic}'...", key=f"chat_{topic_key}")

            if prompt:
                # Add user message to chat history
                st.session_state.chat_histories[topic_key].append({"role": "user", "content": prompt})
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display assistant response
                with st.spinner("Thinking..."):
                    follow_up_answer = ask_follow_up(
                        selected_topic,
                        current_explanation,
                        prompt,
                        API_KEY
                    )
                    if follow_up_answer:
                        st.session_state.chat_histories[topic_key].append({"role": "assistant", "content": follow_up_answer})
                        with st.chat_message("assistant"):
                            st.markdown(follow_up_answer)
                        # No explicit rerun needed here, chat_input triggers it implicitly on submit
                    else:
                        # If API failed, remove the user message to avoid confusion
                        st.session_state.chat_histories[topic_key].pop()
                        st.warning("Couldn't get an answer. Please try asking differently.")

    else:
        st.write(f"Click the button above to get a simple explanation for '{selected_topic}'.")

else:
    st.info("Select a topic from the dropdown menu to get started.")

# --- Footer or additional info ---
st.sidebar.info("Powered by OpenRouter & Gemini Flash 1.5")