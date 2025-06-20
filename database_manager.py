# database_manager.py
# Tool for managing the advanced vector database

import argparse
import os
from src.vector_database import AdvancedVectorDatabase
from src.embeddings import EmbeddingGenerator
from src.preprocessor import DataPreprocessor

class DatabaseManager:
    """Command-line tool for managing vector database"""
    
    def __init__(self, db_path: str = "rag_vector_database"):
        self.db_path = db_path
        self.vector_db = None
        self.embedder = None
    
    def initialize(self):
        """Initialize database and embedder"""
        try:
            self.vector_db = AdvancedVectorDatabase(self.db_path)
            self.embedder = EmbeddingGenerator()
            print(f"✅ Database manager initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def add_file(self, file_path: str, chunk_strategy: str = "lines", chunk_size: int = 200):
        """Add documents from file to database"""
        if not self.vector_db or not self.embedder:
            print("❌ Database not initialized")
            return
        
        try:
            print(f"📚 Adding file: {file_path}")
            
            # Process file
            preprocessor = DataPreprocessor(chunk_size=chunk_size)
            chunks, stats = preprocessor.process_file(file_path, chunk_strategy)
            
            # Generate embeddings
            embeddings = self.embedder.generate_batch_embeddings(chunks)
            
            # Create metadata
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_id': i,
                    'source_file': file_path,
                    'chunk_strategy': chunk_strategy,
                    'chunk_size': chunk_size,
                    'chunk_length': len(chunk),
                    'word_count': len(chunk.split())
                }
                metadata_list.append(metadata)
            
            # Add to database
            self.vector_db.add_documents(chunks, embeddings, metadata_list)
            
            print(f"✅ Successfully added {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            print(f"❌ Error adding file: {e}")
    
    def search(self, query: str, top_k: int = 5):
        """Search database with query"""
        if not self.vector_db or not self.embedder:
            print("❌ Database not initialized")
            return
        
        try:
            print(f"🔍 Searching: {query}")
            
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return
            
            # Search
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            print(f"\n📊 Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"\n{i+1}. Similarity: {result['similarity']:.3f}")
                print(f"   Text: {result['text']}")
                print(f"   Source: {result['metadata']['source_file']}")
                print(f"   Chunk ID: {result['metadata']['chunk_id']}")
        
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    def stats(self):
        """Show database statistics"""
        if not self.vector_db:
            print("❌ Database not initialized")
            return
        
        try:
            stats = self.vector_db.get_statistics()
            print(f"\n📊 Database Statistics:")
            print(f"=" * 30)
            for key, value in stats.items():
                print(f"{key:25}: {value}")
        
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def export(self, export_path: str):
        """Export database to file"""
        if not self.vector_db:
            print("❌ Database not initialized")
            return
        
        try:
            self.vector_db.export_data(export_path)
            print(f"✅ Database exported to {export_path}")
        
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def clear(self):
        """Clear database"""
        if not self.vector_db:
            print("❌ Database not initialized")
            return
        
        confirm = input("⚠️ Are you sure you want to clear the database? (yes/no): ")
        if confirm.lower() == 'yes':
            self.vector_db.clear_database()
            print("✅ Database cleared")
        else:
            print("❌ Operation cancelled")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Vector Database Manager")
    parser.add_argument("--db-path", default="rag_vector_database", help="Database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add file command
    add_parser = subparsers.add_parser("add", help="Add file to database")
    add_parser.add_argument("file", help="File path to add")
    add_parser.add_argument("--strategy", default="lines", choices=["lines", "sentences", "words", "adaptive"])
    add_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search database")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database")
    export_parser.add_argument("output", help="Output file path")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear database")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = DatabaseManager(args.db_path)
    if not manager.initialize():
        return
    
    # Execute command
    if args.command == "add":
        manager.add_file(args.file, args.strategy, args.chunk_size)
    elif args.command == "search":
        manager.search(args.query, args.top_k)
    elif args.command == "stats":
        manager.stats()
    elif args.command == "export":
        manager.export(args.output)
    elif args.command == "clear":
        manager.clear()

if __name__ == "__main__":
    # If no command line arguments, run interactive mode
    import sys
    if len(sys.argv) == 1:
        print("🎮 Interactive Database Manager")
        print("=" * 30)
        
        manager = DatabaseManager()
        if manager.initialize():
            while True:
                print(f"\nAvailable commands:")
                print(f"  1. search <query>")
                print(f"  2. stats")
                print(f"  3. add <file>")
                print(f"  4. export <file>")
                print(f"  5. quit")
                
                command = input(f"\nEnter command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.startswith('search '):
                    query = command[7:]
                    manager.search(query)
                elif command == 'stats':
                    manager.stats()
                elif command.startswith('add '):
                    file_path = command[4:]
                    manager.add_file(file_path)
                elif command.startswith('export '):
                    export_path = command[7:]
                    manager.export(export_path)
                else:
                    print("❌ Unknown command")
    else:
        main()