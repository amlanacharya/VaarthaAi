import os
import json
from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialRAG:
    """
    Retrieval Augmented Generation system for financial regulations and knowledge.
    """

    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize the RAG system.

        Args:
            persist_directory: Directory to persist the vector database.
        """
        self.persist_directory = persist_directory
        # Use HuggingFace sentence-transformers for embeddings (local model)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Use GROQ with Llama model
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize or load the vector database
        try:
            # First check if the directory exists
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                try:
                    self.vectordb = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self.embeddings
                    )
                    logger.info(f"Loaded vector database with {self.vectordb._collection.count()} documents")
                except Exception as e:
                    logger.warning(f"Error loading existing vector database: {e}")
                    logger.info("Recreating the vector database from scratch")
                    # If there's an error, delete the directory and create a new one
                    import shutil
                    shutil.rmtree(persist_directory, ignore_errors=True)
                    os.makedirs(persist_directory, exist_ok=True)
                    self.vectordb = None
            else:
                logger.info("Vector database directory does not exist or is empty. Will create new database.")
                self.vectordb = None
        except Exception as e:
            logger.warning(f"Could not initialize vector database: {e}")
            self.vectordb = None

    def initialize_knowledge_base(self, documents_path: str = "data/regulations"):
        """
        Initialize the knowledge base with financial regulations.

        Args:
            documents_path: Path to the directory containing regulation documents.
        """
        # Create directory if it doesn't exist
        os.makedirs(documents_path, exist_ok=True)

        # Check if we have any documents
        if not os.path.exists(documents_path) or not os.listdir(documents_path):
            logger.warning(f"No documents found in {documents_path}. Creating sample data.")
            self._create_sample_regulations(documents_path)

        # Load documents
        documents = []
        for filename in os.listdir(documents_path):
            if filename.endswith(".json"):
                file_path = os.path.join(documents_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Process each section as a separate document
                    for section in data.get("sections", []):
                        doc = Document(
                            page_content=section.get("content", ""),
                            metadata={
                                "title": data.get("title", ""),
                                "regulation_type": data.get("regulation_type", ""),
                                "section": section.get("section", ""),
                                "section_title": section.get("title", ""),
                                "source": file_path
                            }
                        )
                        documents.append(doc)

        # Create or update the vector database
        if not documents:
            logger.warning("No documents to add to the vector database")
            return

        if self.vectordb is None:
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectordb.add_documents(documents)

        # No need to call persist() with newer versions of Chroma as it's done automatically
        logger.info(f"Vector database initialized with {len(documents)} documents")

    def _create_sample_regulations(self, documents_path: str):
        """
        Create sample regulation documents for testing.

        Args:
            documents_path: Path to save the sample documents.
        """
        # Sample GST regulations
        gst_regulations = {
            "title": "Goods and Services Tax (GST) Regulations",
            "regulation_type": "gst",
            "sections": [
                {
                    "section": "Section 1",
                    "title": "Introduction to GST",
                    "content": "The Goods and Services Tax (GST) is an indirect tax levied on the supply of goods and services in India. It replaced multiple taxes levied by the central and state governments. GST is a comprehensive, multi-stage, destination-based tax that is levied on every value addition."
                },
                {
                    "section": "Section 2",
                    "title": "GST Registration",
                    "content": "Every business whose turnover exceeds Rs. 20 lakhs (Rs. 10 lakhs for North Eastern and hill states) must register for GST. Registration can be done online through the GST portal. Documents required include PAN, business registration documents, and bank account details."
                },
                {
                    "section": "Section 3",
                    "title": "GST Returns Filing",
                    "content": "Regular taxpayers must file monthly or quarterly returns depending on their turnover. The main returns are GSTR-1 (outward supplies), GSTR-3B (summary return), and GSTR-9 (annual return). Returns must be filed even if there is no business activity."
                },
                {
                    "section": "Section 4",
                    "title": "Input Tax Credit",
                    "content": "Input Tax Credit (ITC) is the credit a business can claim for GST paid on purchase of goods or services used for business purposes. To claim ITC, the supplier must have filed their returns and the recipient must have the tax invoice."
                }
            ]
        }

        # Sample Income Tax regulations
        income_tax_regulations = {
            "title": "Income Tax Regulations",
            "regulation_type": "income_tax",
            "sections": [
                {
                    "section": "Section 1",
                    "title": "Introduction to Income Tax",
                    "content": "Income Tax is a direct tax levied on the income earned by individuals, businesses, and other entities. In India, income tax is governed by the Income Tax Act, 1961. The tax is administered by the Central Board of Direct Taxes (CBDT)."
                },
                {
                    "section": "Section 2",
                    "title": "Tax Slabs and Rates",
                    "content": "Income tax in India is levied at progressive rates. For individuals, the tax slabs for FY 2023-24 under the new tax regime are: No tax up to Rs. 3 lakhs, 5% for Rs. 3-6 lakhs, 10% for Rs. 6-9 lakhs, 15% for Rs. 9-12 lakhs, 20% for Rs. 12-15 lakhs, and 30% for income above Rs. 15 lakhs."
                },
                {
                    "section": "Section 3",
                    "title": "Tax Deductions",
                    "content": "The Income Tax Act provides for various deductions under Chapter VI-A (Sections 80C to 80U). These include deductions for investments in PPF, ELSS, life insurance premiums (80C), health insurance premiums (80D), interest on education loans (80E), and interest on home loans (24B)."
                },
                {
                    "section": "Section 4",
                    "title": "TDS Provisions",
                    "content": "Tax Deducted at Source (TDS) is a system where the person making specified payments deducts tax at source and deposits it with the government. Common TDS provisions include TDS on salary (Section 192), interest (Section 194A), professional fees (Section 194J), and rent (Section 194I)."
                }
            ]
        }

        # Save sample regulations
        with open(os.path.join(documents_path, "gst_regulations.json"), 'w', encoding='utf-8') as f:
            json.dump(gst_regulations, f, indent=2)

        with open(os.path.join(documents_path, "income_tax_regulations.json"), 'w', encoding='utf-8') as f:
            json.dump(income_tax_regulations, f, indent=2)

        logger.info("Created sample regulation documents")

    def query(self, query: str, regulation_type: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
        """
        Query the RAG system for financial information.

        Args:
            query: The query string.
            regulation_type: Optional filter for regulation type.
            k: Number of documents to retrieve.

        Returns:
            Dictionary with response and sources.
        """
        if self.vectordb is None:
            self.initialize_knowledge_base()
            if self.vectordb is None:
                return {
                    "response": "Knowledge base not initialized. Please add documents first.",
                    "sources": []
                }

        # Create search filter if regulation_type is provided
        search_filter = None
        if regulation_type:
            search_filter = {"regulation_type": regulation_type}

        # Retrieve relevant documents
        retriever = self.vectordb.as_retriever(
            search_kwargs={"k": k, "filter": search_filter}
        )

        # Create the RAG chain
        prompt_template = """
        You are a financial assistant for Indian businesses and chartered accountants.
        Use the following context to answer the question. If you don't know the answer, say you don't know.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Execute the chain
        response = rag_chain.invoke(query)

        # Get the retrieved documents for sources
        docs = retriever.get_relevant_documents(query)
        sources = [
            {
                "title": doc.metadata.get("title", ""),
                "section": doc.metadata.get("section", ""),
                "section_title": doc.metadata.get("section_title", ""),
                "regulation_type": doc.metadata.get("regulation_type", ""),
                "source": doc.metadata.get("source", "")
            }
            for doc in docs
        ]

        return {
            "response": response,
            "sources": sources
        }
