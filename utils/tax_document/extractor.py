import PyPDF2
import logging
import concurrent.futures
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDF documents with parallel processing."""
    
    def extract(self, pdf_path, max_workers=4):
        """
        Extract text from PDF with parallel page processing.
        
        Args:
            pdf_path: Path to the PDF file
            max_workers: Maximum number of parallel workers
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                logger.info(f"Extracting text from {total_pages} pages using {max_workers} workers")
                
                # Process pages in parallel for large documents
                if total_pages > 20:
                    text_parts = [""] * total_pages
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Create tasks for each page
                        future_to_page = {
                            executor.submit(self._extract_page, reader, page_num): page_num 
                            for page_num in range(total_pages)
                        }
                        
                        # Process results as they complete
                        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=total_pages, desc="Extracting pages"):
                            page_num = future_to_page[future]
                            try:
                                text_parts[page_num] = future.result()
                            except Exception as e:
                                logger.error(f"Error extracting page {page_num}: {e}")
                                text_parts[page_num] = f"[ERROR EXTRACTING PAGE {page_num}]"
                    
                    # Join all parts
                    full_text = "\n".join(text_parts)
                else:
                    # For smaller documents, process sequentially
                    full_text = ""
                    for page_num in tqdm(range(total_pages), desc="Extracting pages"):
                        full_text += self._extract_page(reader, page_num) + "\n"
                        
                return full_text
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _extract_page(self, reader, page_num):
        """Extract text from a single page."""
        page = reader.pages[page_num]
        return page.extract_text()