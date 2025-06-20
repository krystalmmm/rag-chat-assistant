# src/vector_database.py
# Advanced Vector Database Implementation (Step 4)

import json
import pickle
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

class AdvancedVectorDatabase:
    """
    Advanced vector database with persistence, indexing, and metadata management
    """
    
    def __init__(self, db_path: str = "vector_db", embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        
        # Database files
        self.documents_file = os.path.join(db_path, "documents.json")
        self.embeddings_file = os.path.join(db_path, "embeddings.npy")
        self.metadata_file = os.path.join(db_path, "metadata.json")
        self.index_file = os.path.join(db_path, "index.pkl")
        
        # In-memory storage
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.document_index = {}  # For fast lookup
        
        # Load existing data
        self._load_database()
        
        print(f"📚 Vector Database initialized")
        print(f"   Path: {db_path}")
        print(f"   Documents: {len(self.documents)}")
        print(f"   Embedding dimension: {embedding_dim}")
    
    def _generate_doc_id(self, text: str) -> str:
        """Generate unique ID for document"""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def _load_database(self):
        """Load existing database from disk"""
        try:
            # Load documents
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            # Load embeddings
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            # Load index
            if os.path.exists(self.index_file):
                with open(self.index_file, 'rb') as f:
                    self.document_index = pickle.load(f)
            
            # Rebuild index if needed
            if len(self.document_index) != len(self.documents):
                self._rebuild_index()
            
            if self.documents:
                print(f"✅ Loaded existing database with {len(self.documents)} documents")
        
        except Exception as e:
            print(f"⚠️ Error loading database: {e}")
            print("🔄 Starting with empty database")
            self._initialize_empty_database()
    
    def _initialize_empty_database(self):
        """Initialize empty database structures"""
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.document_index = {}
    
    def _rebuild_index(self):
        """Rebuild document index for fast lookup"""
        self.document_index = {}
        for i, doc in enumerate(self.documents):
            doc_id = self._generate_doc_id(doc)
            self.document_index[doc_id] = i
        print("🔄 Rebuilt document index")
    
    def _save_database(self):
        """Save database to disk"""
        try:
            # Save documents
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Save index
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.document_index, f)
            
            print(f"💾 Database saved to {self.db_path}")
        
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def add_documents(self, texts: List[str], embeddings: List[np.ndarray], 
                     metadata: Optional[List[Dict]] = None, batch_size: int = 100):
        """
        Add multiple documents with their embeddings
        
        Args:
            texts: List of document texts
            embeddings: List of corresponding embeddings
            metadata: Optional metadata for each document
            batch_size: Size of batches for processing
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        print(f"📚 Adding {len(texts)} documents to database...")
        
        added_count = 0
        skipped_count = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_texts)
            
            for text, embedding, meta in zip(batch_texts, batch_embeddings, batch_metadata):
                if embedding is None:
                    skipped_count += 1
                    continue
                
                doc_id = self._generate_doc_id(text)
                
                # Check if document already exists
                if doc_id in self.document_index:
                    skipped_count += 1
                    continue
                
                # Add document
                self.documents.append(text)
                
                # Add embedding
                if self.embeddings is None:
                    self.embeddings = embedding.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
                
                # Add metadata
                doc_metadata = {
                    'doc_id': doc_id,
                    'added_at': datetime.now().isoformat(),
                    'length': len(text),
                    'word_count': len(text.split()),
                    **meta
                }
                self.metadata.append(doc_metadata)
                
                # Update index
                self.document_index[doc_id] = len(self.documents) - 1
                
                added_count += 1
        
        print(f"✅ Added {added_count} documents")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} documents (duplicates or invalid embeddings)")
        
        # Save to disk
        self._save_database()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with similarity scores
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to allow for filtering
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            # Apply metadata filter if provided
            if filter_metadata:
                doc_metadata = self.metadata[idx]
                if not all(doc_metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            result = {
                'text': self.documents[idx],
                'similarity': float(similarities[idx]),
                'metadata': self.metadata[idx].copy(),
                'index': int(idx)
            }
            results.append(result)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by its ID"""
        if doc_id in self.document_index:
            idx = self.document_index[doc_id]
            return {
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'embedding': self.embeddings[idx] if self.embeddings is not None else None
            }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        if doc_id not in self.document_index:
            return False
        
        idx = self.document_index[doc_id]
        
        # Remove from lists
        del self.documents[idx]
        del self.metadata[idx]
        
        # Remove from embeddings
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
        
        # Rebuild index
        self._rebuild_index()
        
        # Save to disk
        self._save_database()
        
        print(f"🗑️ Deleted document {doc_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'database_path': self.db_path,
            'disk_usage_mb': self._get_disk_usage(),
        }
        
        if self.documents:
            text_lengths = [len(doc) for doc in self.documents]
            stats.update({
                'avg_document_length': np.mean(text_lengths),
                'min_document_length': np.min(text_lengths),
                'max_document_length': np.max(text_lengths),
            })
        
        return stats
    
    def _get_disk_usage(self) -> float:
        """Calculate disk usage in MB"""
        total_size = 0
        for file_path in [self.documents_file, self.embeddings_file, 
                         self.metadata_file, self.index_file]:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)  # Convert to MB
    
    def export_data(self, export_path: str):
        """Export database to JSON format"""
        
        # Convert statistics to JSON-serializable format
        stats = self.get_statistics()
        json_serializable_stats = {}
        for key, value in stats.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                json_serializable_stats[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                json_serializable_stats[key] = float(value)
            else:
                json_serializable_stats[key] = value
        
        # Convert metadata to JSON-serializable format
        json_serializable_metadata = []
        for meta in self.metadata:
            json_meta = {}
            for key, value in meta.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    json_meta[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    json_meta[key] = float(value)
                else:
                    json_meta[key] = value
            json_serializable_metadata.append(json_meta)
        
        export_data = {
            'documents': self.documents,
            'metadata': json_serializable_metadata,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'statistics': json_serializable_stats,
            'export_timestamp': datetime.now().isoformat(),
            'database_version': '1.0'
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"📤 Database exported to {export_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            # Fallback: export without embeddings
            try:
                export_data_fallback = {
                    'documents': self.documents,
                    'metadata': json_serializable_metadata,
                    'embeddings': None,  # Skip embeddings if they cause issues
                    'statistics': json_serializable_stats,
                    'export_timestamp': datetime.now().isoformat(),
                    'note': 'Embeddings excluded due to serialization issues'
                }
                
                fallback_path = export_path.replace('.json', '_no_embeddings.json')
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data_fallback, f, ensure_ascii=False, indent=2)
                
                print(f"📤 Fallback export (without embeddings) saved to {fallback_path}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback export also failed: {fallback_error}")
    
    def clear_database(self):
        """Clear all data from database"""
        self._initialize_empty_database()
        
        # Remove files
        for file_path in [self.documents_file, self.embeddings_file, 
                         self.metadata_file, self.index_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        print("🗑️ Database cleared")

def test_advanced_vector_db():
    """Test the advanced vector database"""
    print("🧪 Testing Advanced Vector Database")
    print("=" * 50)
    
    # Initialize database
    db = AdvancedVectorDatabase("test_vector_db")
    
    # Sample data
    sample_texts = [
        "Cats can run up to 31 mph over short distances.",
        "Dogs are loyal companions and excellent guards.",
        "Birds have the ability to fly using their wings.",
        "Fish live underwater and breathe through gills.",
        "Elephants are the largest land mammals on Earth."
    ]
    
    # Generate dummy embeddings (normally you'd use real embeddings)
    sample_embeddings = [np.random.rand(768) for _ in sample_texts]
    
    # Add metadata
    sample_metadata = [
        {'category': 'cats', 'topic': 'speed'},
        {'category': 'dogs', 'topic': 'behavior'},
        {'category': 'birds', 'topic': 'flight'},
        {'category': 'fish', 'topic': 'habitat'},
        {'category': 'elephants', 'topic': 'size'}
    ]
    
    # Add documents
    db.add_documents(sample_texts, sample_embeddings, sample_metadata)
    
    # Test search
    query_embedding = np.random.rand(768)
    results = db.search(query_embedding, top_k=3)
    
    print(f"\n🔍 Search Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"     Text: {result['text'][:60]}...")
        print(f"     Category: {result['metadata']['category']}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Advanced Vector Database test completed!")
    
    return db

if __name__ == "__main__":
    test_advanced_vector_db()