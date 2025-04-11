# utils/tax_document/parser.py

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TaxSection:
    """Represents a section of the Income Tax Act."""
    section_number: str
    title: str
    content: str
    chapter: Optional[str] = None
    subsections: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SectionParser:
    """Parse sections from Income Tax Act text."""
    
    def parse(self, text: str) -> List[TaxSection]:
        """
        Extract individual sections from the Income Tax Act text.
        
        Args:
            text: Full text of the Income Tax Act
            
        Returns:
            List of TaxSection objects
        """
        logger.info("Parsing sections from Income Tax Act text")
        
        # First, identify chapters - the format in your document is "CHAPTER X"
        chapter_pattern = r"CHAPTER\s+([IVX]+[A-Z]?)\s*\n(.*?)(?=\n|$)"
        chapters = re.finditer(chapter_pattern, text, re.MULTILINE)
        
        chapter_positions = {}
        chapter_titles = {}
        
        for match in chapters:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            chapter_positions[match.start()] = chapter_num
            chapter_titles[chapter_num] = chapter_title
            logger.info(f"Found chapter: {chapter_num} - {chapter_title}")
        
        # Sort chapter positions
        sorted_chapter_pos = sorted(chapter_positions.keys())
        
        # Now find sections - the format is "N. Title." at the beginning of a line
        section_pattern = r"^(\d+[A-Z]?)\.\s+(.*?)\.(?=\s*$|\s*\n)"
        
        sections = []
        current_chapter = None
        
        # Process each line to find sections
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            # Find the current chapter based on line position
            line_pos = text.find(line)
            for j in range(len(sorted_chapter_pos)):
                if j < len(sorted_chapter_pos) - 1:
                    if sorted_chapter_pos[j] <= line_pos < sorted_chapter_pos[j+1]:
                        current_chapter = chapter_positions[sorted_chapter_pos[j]]
                        break
                else:  # Last chapter
                    if line_pos >= sorted_chapter_pos[j]:
                        current_chapter = chapter_positions[sorted_chapter_pos[j]]
                        break
            
            # Check if this line starts a new section
            section_match = re.match(section_pattern, line)
            
            if section_match:
                # If we were already collecting a section, save it
                if current_section is not None:
                    section_text = '\n'.join(section_content)
                    current_section.content = section_text
                    sections.append(current_section)
                    section_content = []
                
                # Start a new section
                section_num = section_match.group(1)
                section_title = section_match.group(2).strip()
                
                current_section = TaxSection(
                    section_number=section_num,
                    title=section_title,
                    content=line,  # Will be updated as we collect more lines
                    chapter=current_chapter,
                    metadata={
                        "chapter_title": chapter_titles.get(current_chapter, ""),
                        "section_full": f"Section {section_num} - {section_title}"
                    }
                )
                section_content = [line]
            elif current_section is not None:
                # Continue collecting content for the current section
                section_content.append(line)
        
        # Don't forget to add the last section
        if current_section is not None:
            section_text = '\n'.join(section_content)
            current_section.content = section_text
            sections.append(current_section)
        
        # Extract subsections for each section
        for section in sections:
            self._extract_subsections(section)
            
            # Log progress
            if len(sections) % 50 == 0:
                logger.info(f"Processed {len(sections)} sections...")
        
        logger.info(f"Total sections extracted: {len(sections)}")
        return sections
    
    def _extract_subsections(self, section: TaxSection) -> None:
        """Extract subsections from a section."""
        # In the Income Tax Act format, subsections often look like "(1) Text" or "(2A) Text"
        subsection_pattern = r"\((\d+[A-Za-z]?)\)(.*?)(?=\(\d+[A-Za-z]?\)|$)"
        
        subsections = []
        for match in re.finditer(subsection_pattern, section.content, re.DOTALL):
            subsection_num = match.group(1)
            subsection_content = match.group(2).strip()
            
            subsections.append({
                "number": subsection_num,
                "content": subsection_content
            })
        
        section.subsections = subsections
        
        # Update metadata
        section.metadata["subsection_count"] = len(subsections)
        
        # Check for special sections like definitions or explanations
        section_lower = section.content.lower()
        if "explanation" in section_lower:
            section.metadata["has_explanation"] = True
        if "definition" in section_lower or "means" in section_lower:
            section.metadata["is_definition"] = True