# Finsum: Demystifying Company Finances with AI 📊🤖

## 🌟 Project Overview

Finsum is an innovative AI-powered platform designed to break down the complex world of financial statements and make company insights accessible to everyone.

### The Challenge 📈

Financial documents are notoriously complex:
- Filled with technical jargon
- Laden with intricate calculations
- Intimidating for those without a finance background

### Our Solution 🧠

Finsum leverages cutting-edge AI to transform dense financial data into clear, digestible insights that anyone can understand.

## 🚀 Key Features

### 1. Company Story Explorer 
- Select a company and get an AI-generated narrative of its financial journey
- Interactive financial metric visualizations
- Contextual AI chat to answer your specific questions

### 2. Finance Basics Learner
- Explore financial terms and concepts
- Receive beginner-friendly explanations
- Learn through simple analogies and plain language

## 🛠 Technology Stack

### Frontend
- Streamlit

### Backend & Data Processing
- Python
- Pandas, NumPy
- Web Scraping Tools (Selenium, BeautifulSoup)
- PDF Processing (PyMuPDF)

### AI & Machine Learning
- OpenAI SDK (via OpenRouter)
- Langchain
- Transformers
- Sentence-Transformers

### Data Storage
- Pinecone (Vector Database)
- PostgreSQL
- Local File Storage

### Visualization
- Plotly
- Matplotlib
- Seaborn

## 📦 Project Structure

```
finsum/
│
├── .streamlit/           # Streamlit configuration
├── data/                 # Processed company data
├── data_extraction/      # Scripts for raw data retrieval
├── data_preprocessing/   # Data cleaning and formatting scripts
├── pages/                # Streamlit application pages
├── pipeline/             # Intermediate data processing
├── prompts/              # LLM prompt templates
├── utils/                # Utility scripts
├── Welcome.py            # Main Streamlit entry point
└── requirements.txt      # Project dependencies
```

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- Git
- Chrome Browser (for web scraping)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finsum.git
   cd finsum
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API Keys:
   - OpenRouter API Key (for AI features)
   - Pinecone API Key (for vector database)
   
   Store these in `.streamlit/secrets.toml` or `.env`

5. Install ChromeDriver:
   - Download matching your Chrome version
   - Update path in `data_extraction/extract_latest_news.py`

## 🚀 Running the Application

```bash
streamlit run Welcome.py
```

