from typing import List, Tuple
import re

class TextChunker:
    """Handles text chunking strategies for RAG systems"""
    
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_lines(self, lines: List[str]) -> List[str]:
        """
        Simple chunking: each line becomes a chunk
        Good for datasets where each line is a complete fact/statement
        """
        chunks = []
        for i, line in enumerate(lines):
            if len(line.strip()) > 0:
                chunks.append(line.strip())
        
        print(f"📦 Created {len(chunks)} chunks (line-based)")
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks"""
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        print(f"📦 Created {len(chunks)} chunks (sentence-based)")
        return chunks
    
    def chunk_by_words(self, text: str) -> List[str]:
        """Split text into word-based chunks with overlap"""
        words = text.split()
        chunks = []
        
        chunk_size_words = self.chunk_size // 5  # Approximate words per chunk
        overlap_words = self.overlap // 5
        
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            chunk = " ".join(chunk_words)
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Break if we've processed all words
            if i + chunk_size_words >= len(words):
                break
        
        print(f"📦 Created {len(chunks)} chunks (word-based)")
        return chunks
    
    def adaptive_chunk(self, lines: List[str]) -> List[str]:
        """
        Adaptive chunking: choose strategy based on line length
        """
        chunks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # If line is short enough, keep as single chunk
            if len(line) <= self.chunk_size:
                chunks.append(line)
            else:
                # Split long lines using sentence-based chunking
                sub_chunks = self.chunk_by_sentences(line)
                chunks.extend(sub_chunks)
        
        print(f"📦 Created {len(chunks)} chunks (adaptive)")
        return chunks
    
    def analyze_chunks(self, chunks: List[str]) -> dict:
        """Analyze chunk statistics"""
        if not chunks:
            return {"error": "No chunks provided"}
        
        lengths = [len(chunk) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths)
        }
        
        # Find length distribution
        short_chunks = len([l for l in lengths if l < 100])
        medium_chunks = len([l for l in lengths if 100 <= l <= 300])
        long_chunks = len([l for l in lengths if l > 300])
        
        stats["distribution"] = {
            "short (<100 chars)": short_chunks,
            "medium (100-300 chars)": medium_chunks,
            "long (>300 chars)": long_chunks
        }
        
        return stats