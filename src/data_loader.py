import os
import re
from typing import List, Optional

class DataLoader:
    """Handles loading and basic preprocessing of text data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_text_file(self, filename: str, encoding: str = 'utf-8') -> List[str]:
        """Load text file and return list of lines"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            # Remove empty lines and strip whitespace
            lines = [line.strip() for line in lines if line.strip()]
            
            print(f"✅ Loaded {len(lines)} lines from {filename}")
            return lines
            
        except Exception as e:
            raise Exception(f"Error loading file {filename}: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Strip and return
        return text.strip()
    
    def load_and_clean(self, filename: str) -> List[str]:
        """Load file and apply cleaning"""
        raw_lines = self.load_text_file(filename)
        cleaned_lines = [self.clean_text(line) for line in raw_lines]
        
        # Filter out very short lines (less than 10 characters)
        cleaned_lines = [line for line in cleaned_lines if len(line) >= 10]
        
        print(f"📝 After cleaning: {len(cleaned_lines)} valid lines")
        return cleaned_lines