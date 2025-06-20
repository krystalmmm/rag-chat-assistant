# src/preprocessor.py (修复版)
from .data_loader import DataLoader
from .chunker import TextChunker
from typing import List, Dict, Any, Tuple  # 确保 Tuple 被导入

class DataPreprocessor:
    """Main preprocessing pipeline"""
    
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.loader = DataLoader()
        self.chunker = TextChunker(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_file(self, filename: str, chunking_strategy: str = "lines"):
        """
        Complete preprocessing pipeline
        
        Args:
            filename: Input text file
            chunking_strategy: "lines", "sentences", "words", or "adaptive"
        
        Returns:
            Tuple of (chunks, statistics)
        """
        print(f"🔄 Processing {filename} with {chunking_strategy} chunking...")
        
        # Step 1: Load and clean data
        lines = self.loader.load_and_clean(filename)
        
        # Step 2: Apply chunking strategy
        if chunking_strategy == "lines":
            chunks = self.chunker.chunk_by_lines(lines)
        elif chunking_strategy == "sentences":
            # Combine all lines into one text for sentence splitting
            full_text = " ".join(lines)
            chunks = self.chunker.chunk_by_sentences(full_text)
        elif chunking_strategy == "words":
            full_text = " ".join(lines)
            chunks = self.chunker.chunk_by_words(full_text)
        elif chunking_strategy == "adaptive":
            chunks = self.chunker.adaptive_chunk(lines)
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        # Step 3: Analyze results
        stats = self.chunker.analyze_chunks(chunks)
        stats["chunking_strategy"] = chunking_strategy
        stats["original_lines"] = len(lines)
        
        return chunks, stats
    
    def compare_strategies(self, filename: str):
        """Compare different chunking strategies"""
        strategies = ["lines", "sentences", "words", "adaptive"]
        results = {}
        
        print("🔍 Comparing chunking strategies...")
        
        for strategy in strategies:
            try:
                chunks, stats = self.process_file(filename, strategy)
                results[strategy] = {
                    "chunks": chunks[:3],  # Store first 3 chunks as examples
                    "stats": stats
                }
                print(f"  ✅ {strategy}: {stats['total_chunks']} chunks")
            except Exception as e:
                print(f"  ❌ {strategy}: Error - {e}")
                results[strategy] = {"error": str(e)}
        
        return results