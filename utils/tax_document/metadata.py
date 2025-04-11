
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetadataGenerator:
    """Generate and save metadata about tax document sections."""
    
    def generate(self, sections, chunked_sections):
        """
        Generate comprehensive metadata for sections.
        
        Args:
            sections: List of section objects
            chunked_sections: List of dictionaries with section chunks
            
        Returns:
            Dictionary with metadata
        """
        logger.info("Generating tax document metadata")
        
        # Create basic metadata structure
        metadata = {
            "section_count": len(sections),
            "sections": [],
            "chapters": {},
            "cross_references": {},
            "categories": self._categorize_sections(sections),
            "assessment_year": "2023-24"  # Default, can be overridden
        }
        
        # Populate sections data
        for section in sections:
            section_data = {
                "section_number": section.section_number,
                "title": section.title,
                "chapter": section.chapter,
                "chapter_title": section.metadata.get("chapter_title", ""),
                "subsection_count": len(section.subsections),
                "chunk_count": self._get_chunk_count(section.section_number, chunked_sections),
                "is_definition": section.metadata.get("is_definition", False),
                "has_explanation": section.metadata.get("has_explanation", False),
                "keywords": self._extract_keywords(section)
            }
            
            metadata["sections"].append(section_data)
            
            # Add to chapters index
            if section.chapter:
                if section.chapter not in metadata["chapters"]:
                    metadata["chapters"][section.chapter] = {
                        "title": section.metadata.get("chapter_title", ""),
                        "sections": []
                    }
                metadata["chapters"][section.chapter]["sections"].append(section.section_number)
        
        # Extract cross-references between sections
        for section in sections:
            references = self._extract_cross_references(section.content)
            if references:
                metadata["cross_references"][section.section_number] = references
        
        # Build table of contents
        metadata["table_of_contents"] = self._build_toc(metadata["chapters"])
        
        return metadata
    
    def _get_chunk_count(self, section_number, chunked_sections):
        """Get the number of chunks for a section."""
        for section_info in chunked_sections:
            if section_info["section_number"] == section_number:
                return len(section_info["chunks"])
        return 0
    
    def _extract_cross_references(self, content):
        """Extract section cross-references from content."""
        # Pattern to detect references to other sections
        reference_pattern = r"section\s+(\d+[A-Z]?(?:-\d+[A-Z]?)?)"
        
        # Find all references
        matches = re.finditer(reference_pattern, content, re.IGNORECASE)
        references = set()
        
        for match in matches:
            references.add(match.group(1))
        
        return list(references)
    
    def _extract_keywords(self, section):
        """Extract key terms from a section."""
        # Common tax terms to look for
        tax_terms = [
            "income", "deduction", "exemption", "assessment", "return",
            "tax", "relief", "rebate", "surcharge", "cess", "penalty", 
            "interest", "refund", "credit", "assessee", "taxable"
        ]
        
        keywords = []
        
        # Check for presence of common tax terms
        for term in tax_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', section.content, re.IGNORECASE):
                keywords.append(term)
        
        # Look for defined terms (often in quotes or after "means")
        defined_terms = re.findall(r'"([^"]+)"\s+means', section.content)
        defined_terms.extend(re.findall(r'term\s+"([^"]+)"', section.content))
        
        keywords.extend(defined_terms)
        
        # Deduplicate and return
        return list(set(keywords))
    
    def _categorize_sections(self, sections):
        """Categorize sections by their purpose."""
        categories = {
            "definitions": [],
            "deductions": [],
            "exemptions": [],
            "calculation": [],
            "procedure": []
        }
        
        for section in sections:
            # Check section title and content to categorize
            content_lower = section.content.lower()
            title_lower = section.title.lower()
            
            if section.metadata.get("is_definition", False) or "definition" in title_lower:
                categories["definitions"].append(section.section_number)
                
            elif "deduction" in content_lower or "deduction" in title_lower:
                categories["deductions"].append(section.section_number)
                
            elif "exempt" in content_lower or "exemption" in title_lower:
                categories["exemptions"].append(section.section_number)
                
            elif any(term in content_lower for term in ["compute", "calculation", "tax rate", "slab"]):
                categories["calculation"].append(section.section_number)
                
            elif any(term in content_lower for term in ["procedure", "return", "assessment", "appeal"]):
                categories["procedure"].append(section.section_number)
        
        return categories
    
    def _build_toc(self, chapters):
        """Build a hierarchical table of contents without sorting."""
        toc = []
        
        # Skip sorting and just use chapters in original order
        for chapter_id in chapters.keys():
            chapter = chapters[chapter_id]
            
            chapter_entry = {
                "id": chapter_id,
                "title": chapter["title"],
                "sections": chapter["sections"]  # No sorting, just use original order
            }
            
            toc.append(chapter_entry)
        
        return toc
    
    def save_metadata(self, metadata, output_path):
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save the metadata file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved metadata to {output_path}")
