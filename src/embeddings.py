# src/embeddings.py
import ollama
import numpy as np
import time
from typing import List, Optional, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class EmbeddingGenerator:
    """Handles text embedding generation using Ollama"""
    
    def __init__(self, model_name: str = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'):
        self.model_name = model_name
        self.embedding_cache = {}  # Simple in-memory cache
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the embedding model is available"""
        try:
            # Test with a simple text
            test_response = ollama.embed(model=self.model_name, input="test")
            if test_response and 'embeddings' in test_response:
                embedding_dim = len(test_response['embeddings'][0])
                print(f"✅ Embedding model ready: {self.model_name}")
                print(f"📏 Embedding dimension: {embedding_dim}")
                self.embedding_dim = embedding_dim
            else:
                raise Exception("Invalid response from embedding model")
        except Exception as e:
            print(f"❌ Embedding model verification failed: {e}")
            raise Exception(f"Cannot initialize embedding model: {e}")
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            use_cache: Whether to use/store in cache
            
        Returns:
            NumPy array of embedding or None if failed
        """
        if not text or not text.strip():
            print("⚠️ Empty text provided for embedding")
            return None
        
        # Clean text
        text = text.strip()
        
        # Check cache
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = ollama.embed(model=self.model_name, input=text)
            
            if response and 'embeddings' in response and response['embeddings']:
                embedding = np.array(response['embeddings'][0], dtype=np.float32)
                
                # Store in cache
                if use_cache:
                    self.embedding_cache[text] = embedding
                
                return embedding
            else:
                print(f"❌ Invalid embedding response for text: {text[:50]}...")
                return None
                
        except Exception as e:
            print(f"❌ Embedding generation failed for text: {text[:50]}... Error: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str], 
                                batch_size: int = 10, 
                                show_progress: bool = True) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts with batching
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress updates
            
        Returns:
            List of embeddings (some may be None if failed)
        """
        if not texts:
            return []
        
        embeddings = []
        total_texts = len(texts)
        
        if show_progress:
            print(f"🔄 Generating embeddings for {total_texts} texts...")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for j, text in enumerate(batch):
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)
                
                if show_progress and (i + j + 1) % 5 == 0:
                    progress = ((i + j + 1) / total_texts) * 100
                    print(f"   📊 Progress: {progress:.1f}% ({i + j + 1}/{total_texts})")
            
            embeddings.extend(batch_embeddings)
            
            # Small delay to prevent overwhelming the API
            if i + batch_size < total_texts:
                time.sleep(0.1)
        
        successful_embeddings = len([e for e in embeddings if e is not None])
        
        if show_progress:
            print(f"✅ Generated {successful_embeddings}/{total_texts} embeddings successfully")
        
        return embeddings
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in valid range
        return float(np.clip(similarity, -1.0, 1.0))
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray],
                         candidate_texts: List[str],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar texts to query based on embeddings
        
        Args:
            query_embedding: Embedding of the query text
            candidate_embeddings: List of candidate embeddings
            candidate_texts: List of candidate texts (same order as embeddings)
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with text, embedding, and similarity score
        """
        if query_embedding is None:
            return []
        
        similarities = []
        
        for i, (candidate_embedding, text) in enumerate(zip(candidate_embeddings, candidate_texts)):
            if candidate_embedding is not None:
                similarity = self.cosine_similarity(query_embedding, candidate_embedding)
                similarities.append({
                    'text': text,
                    'embedding': candidate_embedding,
                    'similarity': similarity,
                    'index': i
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache"""
        return {
            'cache_size': len(self.embedding_cache),
            'model_name': self.model_name,
            'embedding_dimension': getattr(self, 'embedding_dim', 'unknown')
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        print(f"🗑️ Cleared embedding cache ({cache_size} entries)")

class VectorDatabase:
    """Simple in-memory vector database for storing and searching embeddings"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.documents = []  # List of document texts
        self.embeddings = []  # List of corresponding embeddings
        self.metadata = []   # List of metadata for each document
    
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector database
        
        Args:
            texts: List of document texts
            metadata: Optional metadata for each document
        """
        if not texts:
            return
        
        print(f"📚 Adding {len(texts)} documents to vector database...")
        
        # Generate embeddings
        new_embeddings = self.embedding_generator.generate_batch_embeddings(texts)
        
        # Add to database
        for i, (text, embedding) in enumerate(zip(texts, new_embeddings)):
            if embedding is not None:
                self.documents.append(text)
                self.embeddings.append(embedding)
                
                # Add metadata
                if metadata and i < len(metadata):
                    self.metadata.append(metadata[i])
                else:
                    self.metadata.append({'index': len(self.documents) - 1})
        
        print(f"✅ Added {len([e for e in new_embeddings if e is not None])} documents successfully")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of search results with text, similarity, and metadata
        """
        if not self.documents:
            print("⚠️ No documents in database")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        if query_embedding is None:
            print("❌ Failed to generate query embedding")
            return []
        
        # Find similar documents
        results = self.embedding_generator.find_most_similar(
            query_embedding, self.embeddings, self.documents, top_k
        )
        
        # Add metadata to results
        for result in results:
            doc_index = result['index']
            if doc_index < len(self.metadata):
                result['metadata'] = self.metadata[doc_index]
            else:
                result['metadata'] = {}
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_documents': len(self.documents),
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': getattr(self.embedding_generator, 'embedding_dim', 'unknown'),
            'cache_stats': self.embedding_generator.get_cache_stats()
        }

def test_embeddings():
    """Test the embedding functionality"""
    print("🧪 Testing Embedding System")
    print("=" * 40)
    
    try:
        # Initialize embedding generator
        embedder = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "Cats can run up to 31 mph"
        embedding = embedder.generate_embedding(test_text)
        
        if embedding is not None:
            print(f"✅ Single embedding test passed")
            print(f"   Text: {test_text}")
            print(f"   Embedding shape: {embedding.shape}")
            print(f"   Embedding preview: {embedding[:5]}...")
        
        # Test batch embeddings
        test_texts = [
            "Cats can run up to 31 mph",
            "Dogs are loyal companions",
            "Birds can fly in the sky"
        ]
        
        batch_embeddings = embedder.generate_batch_embeddings(test_texts)
        successful_count = len([e for e in batch_embeddings if e is not None])
        
        print(f"\n✅ Batch embedding test: {successful_count}/{len(test_texts)} successful")
        
        # Test similarity
        if len(batch_embeddings) >= 2 and batch_embeddings[0] is not None and batch_embeddings[1] is not None:
            similarity = embedder.cosine_similarity(batch_embeddings[0], batch_embeddings[1])
            print(f"✅ Similarity test: {similarity:.3f}")
        
        # Test vector database
        print(f"\n📚 Testing Vector Database:")
        vector_db = VectorDatabase(embedder)
        
        # Add documents
        documents = [
            "Cats can travel at a top speed of approximately 31 mph over short distances.",
            "A cat's hearing is better than a dog's. Cats can hear high-frequency sounds up to 64,000 Hz.",
            "Cats spend 70% of their lives sleeping, which is 13-16 hours a day."
        ]
        
        vector_db.add_documents(documents)
        
        # Search
        search_results = vector_db.search("How fast can cats move?", top_k=2)
        
        print(f"🔍 Search results:")
        for i, result in enumerate(search_results):
            print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
            print(f"      Text: {result['text'][:60]}...")
        
        print(f"\n📊 Database stats: {vector_db.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

if __name__ == "__main__":
    test_embeddings()