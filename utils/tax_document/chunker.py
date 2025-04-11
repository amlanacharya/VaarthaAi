import os
import json
import logging
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Create and save chunks from tax document sections."""
    
    def __init__(self, chunk_size=768, chunk_overlap=128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_chunks(self, sections, max_workers=4):
        """
        Create chunks from sections for vector storage.
        
        Args:
            sections: List of section objects
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of dictionaries with section info and chunks
        """
        logger.info(f"Creating chunks for {len(sections)} sections with {max_workers} workers")
        
        # Process sections in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create chunking tasks
            future_to_section = {
                executor.submit(self._chunk_section, section): section 
                for section in sections
            }
            
            # Collect results
            results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_section), total=len(sections), desc="Chunking sections"):
                section = future_to_section[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error chunking section {section.section_number}: {e}")
        
        return results
    
    def _chunk_section(self, section):
        """Process a single section into chunks."""
        # Determine appropriate chunk size based on section characteristics
        chunk_size = self.chunk_size
        
        # Use smaller chunks for definitions to keep them more precise
        if section.metadata.get("is_definition", False):
            chunk_size = min(512, chunk_size)
        
        # Use larger chunks for sections with few subsections but long content
        if len(section.subsections) <= 2 and len(section.content) > 1000:
            chunk_size = max(1024, chunk_size)
            
        # Create a section-specific splitter if needed
        if chunk_size != self.chunk_size:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(section.content)
        else:
            chunks = self.splitter.split_text(section.content)
        
        # For sections with subsections, ensure subsection boundaries are respected
        if section.subsections and len(chunks) > 1:
            # Try to keep subsections in their own chunks where possible
            refined_chunks = []
            current_chunk = ""
            
            for subsection in section.subsections:
                subsection_text = f"({subsection['number']}) {subsection['content']}"
                
                # If adding this subsection would exceed chunk size, start a new chunk
                if len(current_chunk) + len(subsection_text) > chunk_size:
                    if current_chunk:
                        refined_chunks.append(current_chunk)
                    current_chunk = subsection_text
                else:
                    current_chunk += "\n\n" + subsection_text if current_chunk else subsection_text
            
            # Add the last chunk if not empty
            if current_chunk:
                refined_chunks.append(current_chunk)
                
            # Use refined chunks if they're not empty
            if refined_chunks:
                chunks = refined_chunks
        
        return {
            "section_number": section.section_number,
            "title": section.title,
            "chapter": section.chapter,
            "metadata": section.metadata,
            "chunks": chunks
        }
    
    def save_chunks(self, chunked_sections, output_dir):
        """
        Save chunks to JSON files.
        
        Args:
            chunked_sections: List of dictionaries with section info and chunks
            output_dir: Directory to save chunk files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for section_info in chunked_sections:
            # Clean section number for filename (replace slashes, etc.)
            safe_section_number = section_info['section_number'].replace('/', '_')
            
            output_file = os.path.join(
                output_dir,
                f"section_{safe_section_number}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(section_info, f, indent=2)