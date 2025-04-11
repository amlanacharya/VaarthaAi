# process_tax_document.py

import os
import argparse
import logging
from utils.tax_document import TaxDocumentPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Process an Income Tax Act document."""
    parser = argparse.ArgumentParser(description="Process Income Tax Act PDF")
    
    parser.add_argument(
        "--pdf", 
        required=True,
        help="Path to the Income Tax Act PDF file"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/income_tax_act",
        help="Directory to store processed data"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int,
        default=768,
        help="Default size of chunks for vectorization"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int,
        default=128,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--assessment-year",
        default="2023-24",
        help="Assessment year for this version"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        return
    
    # Create configuration
    config = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'max_workers': args.workers,
        'assessment_year': args.assessment_year
    }
    
    # Run the pipeline
    try:
        pipeline = TaxDocumentPipeline(args.pdf, args.output_dir, config)
        result = pipeline.run()
        
        logger.info("Tax document processing completed successfully")
        logger.info(f"Processed {result['sections_count']} sections")
        logger.info(f"Created {result['chunks_count']} chunks")
        logger.info(f"Raw text saved to: {result['raw_text_path']}")
        logger.info(f"Metadata saved to: {result['metadata_path']}")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()