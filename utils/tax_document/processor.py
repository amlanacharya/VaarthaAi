
import os
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TaxDocumentPipeline:
    """Pipeline coordinator for tax document processing."""
    
    def __init__(self, document_path, output_dir, config=None):
        self.document_path = document_path
        self.output_dir = output_dir
        self.config = config or {}
        
        # Initialize pipeline components
        from .extractor import PDFExtractor
        from .parser import SectionParser
        from .chunker import DocumentChunker
        from .metadata import MetadataGenerator
        
        self.extractor = PDFExtractor()
        self.parser = SectionParser()
        self.chunker = DocumentChunker(
            chunk_size=self.config.get('chunk_size', 768),
            chunk_overlap=self.config.get('chunk_overlap', 128)
        )
        self.metadata_gen = MetadataGenerator()
    
    def run(self):
        """Run the full processing pipeline."""
        logger.info(f"Running tax document pipeline for {self.document_path}")
        
        # Step 1: Extract text from PDF
        raw_text = self.extractor.extract(self.document_path)
        raw_text_path = os.path.join(self.output_dir, "raw", "income_tax_act_full.txt")
        os.makedirs(os.path.dirname(raw_text_path), exist_ok=True)
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        logger.info(f"Text extracted and saved to {raw_text_path}")
        
        # Step 2: Parse sections
        sections = self.parser.parse(raw_text)
        logger.info(f"Parsed {len(sections)} sections from document")
        
        # Step 3: Generate chunks for vector storage
        chunks = self.chunker.create_chunks(sections)
        chunks_dir = os.path.join(self.output_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        self.chunker.save_chunks(chunks, chunks_dir)
        logger.info(f"Created and saved chunks to {chunks_dir}")
        
        # Step 4: Generate metadata
        metadata = self.metadata_gen.generate(sections, chunks)
        metadata_path = os.path.join(self.output_dir, "metadata", "tax_document_metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        self.metadata_gen.save_metadata(metadata, metadata_path)
        logger.info(f"Generated and saved metadata to {metadata_path}")
        
        return {
            "raw_text_path": raw_text_path,
            "sections_count": len(sections),
            "chunks_count": sum(len(c['chunks']) for c in chunks),
            "metadata_path": metadata_path
        }