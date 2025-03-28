import os
import json
import uuid
import time
import logging
import torch
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# --- Configuration ---
load_dotenv()
CLEANED_JSON_DIR = r'C:\Techkriti\Cleaned_Json'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
PINECONE_BATCH_SIZE = 128
PINECONE_NAMESPACE = "report_sections"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Changed model
EMBEDDING_DIMENSION = 384  # Updated dimension

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ingestion.log'), logging.StreamHandler()]
)

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

class IngestionPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.pinecone = None
        self.index = None
        
    def initialize_models(self):
        """Initialize embedding model and Pinecone connection"""
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
            logging.info(f"Initialized embedding model on {self.device}")
            
            self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
            self._ensure_pinecone_index()
            
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def _ensure_pinecone_index(self):
        """Ensure Pinecone index exists and is ready"""
        if PINECONE_INDEX_NAME not in self.pinecone.list_indexes().names():
            logging.info("Creating new Pinecone index")
            self.pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            self._wait_for_index_ready()
        
        self.index = self.pinecone.Index(PINECONE_INDEX_NAME)
        logging.info("Pinecone index connection established")

    def _wait_for_index_ready(self):
        """Wait for index to become ready"""
        while True:
            try:
                status = self.pinecone.describe_index(PINECONE_INDEX_NAME).status
                if status['ready']:
                    break
                logging.info("Waiting for index initialization...")
                time.sleep(10)
            except Exception as e:
                logging.error(f"Error waiting for index: {str(e)}")
                raise

    @staticmethod
    def parse_metadata_from_filename(filename: str) -> Dict:
        """Enhanced filename parsing with additional metadata fields"""
        try:
            parts = filename.split('_')
            if len(parts) >= 5:
                return {
                    'company_code': parts[2],
                    'company_name': ' '.join(parts[2:-3]).title(),
                    'report_years': f"{parts[-3]}-{parts[-2]}",
                    'report_type': parts[0],
                    'document_category': "Annual Report",  # New metadata field
                    'language': "English"  # New metadata field
                }
            return {}
        except Exception as e:
            logging.warning(f"Filename parsing failed: {str(e)}")
            return {}

    def process_documents(self):
        """Main processing pipeline"""
        documents = self._load_and_chunk_documents()
        self._embed_and_upsert(documents)

    def _load_and_chunk_documents(self) -> List[Dict]:
        """Load and chunk documents with enhanced section handling"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True
        )

        all_chunks = []
        
        for root, _, files in os.walk(CLEANED_JSON_DIR):
            for file in files:
                if file.endswith('_cleaned.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        base_meta = self.parse_metadata_from_filename(data.get("filename", ""))
                        sections = self._process_sections(data)
                        
                        for section_name, content in sections.items():
                            chunks = splitter.split_text(content)
                            for i, chunk in enumerate(chunks):
                                chunk_meta = {
                                    **base_meta,
                                    'section': section_name,
                                    'chunk_index': i,
                                    'source_file': file,
                                    'document_path': file_path,
                                    'content_type': self._get_content_type(section_name)  # New metadata
                                }
                                all_chunks.append({
                                    'text': chunk,
                                    'metadata': self._clean_metadata(chunk_meta)
                                })
                                
                    except Exception as e:
                        logging.error(f"Failed to process {file_path}: {str(e)}")
                        continue

        logging.info(f"Processed {len(all_chunks)} document chunks")
        return all_chunks

    def _process_sections(self, data: Dict) -> Dict:
        """Process all 7 required sections with proper handling"""
        target_sections = [
            'executive_summary',
            'financial_analysis', 
            'risk_factors',
            'management_discussion',
            'operational_review',
            'technical_jargon',
            'press_releases'
        ]
        
        sections = {}
        
        for section in target_sections:
            if section not in data:
                continue
                
            content = []
            for item in data[section]:
                if isinstance(item, dict):
                    # Handle regular sections with text and pages
                    text = item.get('text', '').replace('\n', ' ').strip()
                    if text:
                        content.append(text)
                        if 'page' in item:
                            content.append(f"(Page {item['page']})")
                elif isinstance(item, str):
                    # Handle technical jargon strings
                    cleaned = item.replace('\n', ' ').strip()
                    if cleaned:
                        content.append(cleaned)
                elif isinstance(item, list):
                    # Handle press release attributes
                    pr_text = ' '.join([str(elem) for elem in item]).strip()
                    if pr_text:
                        content.append(pr_text)
            
            if content:
                sections[section] = ' '.join(content)
                
        return sections

    def _get_content_type(self, section_name: str) -> str:
        """Categorize content for metadata"""
        if section_name == 'technical_jargon':
            return 'Technical Terminology'
        elif section_name == 'press_releases':
            return 'Press Release'
        return 'Report Section'

    @staticmethod
    def _clean_metadata(metadata: Dict) -> Dict:
        """Ensure Pinecone-compatible metadata types"""
        cleaned = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned

    def _embed_and_upsert(self, documents: List[Dict]):
        """Batch process embeddings and upsert to Pinecone"""
        total_docs = len(documents)
        logging.info(f"Starting embedding and upsert of {total_docs} chunks")
        
        for i in range(0, total_docs, PINECONE_BATCH_SIZE):
            batch = documents[i:i+PINECONE_BATCH_SIZE]
            try:
                # No prefix needed for all-MiniLM-L6-v2
                texts = [doc['text'] for doc in batch]
                
                embeddings = self.model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    device=self.device,
                    normalize_embeddings=True
                ).tolist()
                
                vectors = []
                for j, doc in enumerate(batch):
                    vectors.append((
                        str(uuid.uuid4()),
                        embeddings[j],
                        doc['metadata']
                    ))
                
                self.index.upsert(
                    vectors=vectors,
                    namespace=PINECONE_NAMESPACE
                )
                logging.info(f"Upserted batch {i//PINECONE_BATCH_SIZE + 1}")
                
            except Exception as e:
                logging.error(f"Failed to process batch {i}: {str(e)}")

        logging.info("Upsert process completed")
        self._log_index_stats()

    def _log_index_stats(self):
        """Log final index statistics"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(PINECONE_NAMESPACE)
            if namespace_stats:
                logging.info(f"Final vector count: {namespace_stats.vector_count}")
            else:
                logging.warning("No vectors found in target namespace")
        except Exception as e:
            logging.error(f"Failed to get index stats: {str(e)}")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    try:
        pipeline.initialize_models()
        pipeline.process_documents()
        logging.info("Ingestion pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")