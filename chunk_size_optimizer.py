# chunk_size_optimizer.py
# 工具来测试和优化 chunk size

import ollama
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class ChunkSizeOptimizer:
    """优化 RAG 系统的 chunk size"""
    
    def __init__(self):
        self.embedding_model = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
        
    def create_chunks(self, text: str, chunk_size: int, overlap: int = 50) -> List[str]:
        """创建不同大小的 chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 尝试在句子边界结束
            if end < len(text):
                # 寻找最近的句号、问号或感叹号
                for i in range(end, max(start, end-50), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def analyze_chunk_sizes(self, documents: List[str], chunk_sizes: List[int]) -> Dict:
        """分析不同 chunk sizes 的效果"""
        results = {}
        
        print("🔍 分析不同 chunk sizes...")
        
        for chunk_size in chunk_sizes:
            print(f"\n📏 测试 chunk size: {chunk_size}")
            
            all_chunks = []
            for doc in documents:
                chunks = self.create_chunks(doc, chunk_size)
                all_chunks.extend(chunks)
            
            # 分析统计数据
            chunk_lengths = [len(chunk) for chunk in all_chunks]
            word_counts = [len(chunk.split()) for chunk in all_chunks]
            
            # 测试 embedding 效果
            embedding_success = 0
            embedding_times = []
            
            for chunk in all_chunks[:5]:  # 测试前5个chunks
                try:
                    import time
                    start_time = time.time()
                    response = ollama.embed(model=self.embedding_model, input=chunk)
                    embedding_time = time.time() - start_time
                    
                    if response and 'embeddings' in response:
                        embedding_success += 1
                        embedding_times.append(embedding_time)
                except Exception as e:
                    print(f"❌ Embedding 失败: {e}")
            
            results[chunk_size] = {
                'total_chunks': len(all_chunks),
                'avg_length': np.mean(chunk_lengths),
                'min_length': np.min(chunk_lengths),
                'max_length': np.max(chunk_lengths),
                'avg_words': np.mean(word_counts),
                'embedding_success_rate': embedding_success / min(5, len(all_chunks)),
                'avg_embedding_time': np.mean(embedding_times) if embedding_times else 0,
                'sample_chunks': all_chunks[:3]  # 保存样本
            }
            
            print(f"   📊 总 chunks: {results[chunk_size]['total_chunks']}")
            print(f"   📏 平均长度: {results[chunk_size]['avg_length']:.1f} 字符")
            print(f"   📝 平均单词数: {results[chunk_size]['avg_words']:.1f}")
            print(f"   ✅ Embedding 成功率: {results[chunk_size]['embedding_success_rate']:.2%}")
        
        return results
    
    def test_retrieval_quality(self, documents: List[str], chunk_sizes: List[int], 
                             test_queries: List[str]) -> Dict:
        """测试不同 chunk sizes 的检索质量"""
        
        print("\n🎯 测试检索质量...")
        retrieval_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"\n📏 测试 chunk size {chunk_size} 的检索效果...")
            
            # 创建 chunks
            all_chunks = []
            for doc in documents:
                chunks = self.create_chunks(doc, chunk_size)
                all_chunks.extend(chunks)
            
            # 为每个查询测试检索
            query_scores = []
            for query in test_queries:
                try:
                    # 获取查询的 embedding
                    query_response = ollama.embed(model=self.embedding_model, input=query)
                    if not query_response or 'embeddings' not in query_response:
                        continue
                    
                    query_embedding = np.array(query_response['embeddings'][0])
                    
                    # 计算与所有 chunks 的相似度
                    similarities = []
                    for chunk in all_chunks[:10]:  # 限制为前10个chunks以节省时间
                        try:
                            chunk_response = ollama.embed(model=self.embedding_model, input=chunk)
                            if chunk_response and 'embeddings' in chunk_response:
                                chunk_embedding = np.array(chunk_response['embeddings'][0])
                                
                                # 计算余弦相似度
                                similarity = np.dot(query_embedding, chunk_embedding) / (
                                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                                )
                                similarities.append(similarity)
                        except:
                            continue
                    
                    if similarities:
                        max_similarity = max(similarities)
                        avg_similarity = np.mean(similarities)
                        query_scores.append({
                            'query': query,
                            'max_similarity': max_similarity,
                            'avg_similarity': avg_similarity
                        })
                        
                        print(f"   🔍 '{query}': 最高相似度 {max_similarity:.3f}")
                
                except Exception as e:
                    print(f"   ❌ 查询 '{query}' 失败: {e}")
            
            retrieval_results[chunk_size] = {
                'query_scores': query_scores,
                'avg_max_similarity': np.mean([qs['max_similarity'] for qs in query_scores]) if query_scores else 0,
                'avg_avg_similarity': np.mean([qs['avg_similarity'] for qs in query_scores]) if query_scores else 0
            }
        
        return retrieval_results
    
    def recommend_chunk_size(self, analysis_results: Dict, retrieval_results: Dict) -> Dict:
        """基于分析结果推荐最佳 chunk size"""
        
        print("\n💡 分析结果和推荐...")
        
        recommendations = {}
        
        for chunk_size, stats in analysis_results.items():
            score = 0
            reasons = []
            
            # 评分标准
            
            # 1. Embedding 成功率 (权重: 30%)
            embedding_score = stats['embedding_success_rate'] * 30
            score += embedding_score
            if stats['embedding_success_rate'] == 1.0:
                reasons.append("✅ Embedding 100% 成功")
            elif stats['embedding_success_rate'] > 0.8:
                reasons.append("⚠️ Embedding 成功率良好")
            else:
                reasons.append("❌ Embedding 成功率较低")
            
            # 2. Chunk 数量合理性 (权重: 20%)
            if 5 <= stats['total_chunks'] <= 20:
                chunk_count_score = 20
                reasons.append("✅ Chunk 数量适中")
            elif stats['total_chunks'] < 5:
                chunk_count_score = 10
                reasons.append("⚠️ Chunk 数量偏少")
            else:
                chunk_count_score = 15
                reasons.append("⚠️ Chunk 数量偏多")
            score += chunk_count_score
            
            # 3. 平均长度合理性 (权重: 25%)
            avg_len = stats['avg_length']
            if 150 <= avg_len <= 300:
                length_score = 25
                reasons.append("✅ 长度适中")
            elif 100 <= avg_len < 150:
                length_score = 20
                reasons.append("⚠️ 长度偏短")
            elif 300 < avg_len <= 500:
                length_score = 20
                reasons.append("⚠️ 长度偏长")
            else:
                length_score = 10
                reasons.append("❌ 长度不合适")
            score += length_score
            
            # 4. 检索质量 (权重: 25%)
            if chunk_size in retrieval_results:
                retrieval_score = retrieval_results[chunk_size]['avg_max_similarity'] * 25
                score += retrieval_score
                if retrieval_results[chunk_size]['avg_max_similarity'] > 0.7:
                    reasons.append("✅ 检索质量优秀")
                elif retrieval_results[chunk_size]['avg_max_similarity'] > 0.5:
                    reasons.append("⚠️ 检索质量良好")
                else:
                    reasons.append("❌ 检索质量一般")
            
            recommendations[chunk_size] = {
                'score': score,
                'reasons': reasons,
                'stats': stats
            }
        
        # 找出最佳 chunk size
        best_chunk_size = max(recommendations.keys(), key=lambda k: recommendations[k]['score'])
        
        print(f"\n🏆 推荐的 chunk size: {best_chunk_size}")
        print(f"📊 综合评分: {recommendations[best_chunk_size]['score']:.1f}/100")
        print("📋 推荐理由:")
        for reason in recommendations[best_chunk_size]['reasons']:
            print(f"   {reason}")
        
        return {
            'recommended_size': best_chunk_size,
            'all_scores': recommendations,
            'best_score': recommendations[best_chunk_size]['score']
        }

def test_cat_facts_chunking():
    """测试 cat facts 的最佳 chunking 策略"""
    
    print("🐱 Cat Facts Chunk Size 优化")
    print("="*50)
    
    # Cat facts 数据
    cat_facts = [
        "Cats can travel at a top speed of approximately 31 mph over short distances.",
        "A cat's hearing is better than a dog's. Cats can hear high-frequency sounds up to 64,000 Hz.",
        "Cats spend 70% of their lives sleeping, which is 13-16 hours a day.",
        "A group of cats is called a clowder, and a group of kittens is called a kindle.",
        "Cats have five toes on their front paws but only four on their back paws.",
        "A cat's purr vibrates at a frequency of 25-50 Hz, which can promote healing.",
        "The oldest known pet cat existed 9,500 years ago in Cyprus.",
        "Cats can make over 100 different vocal sounds, while dogs can only make 10.",
        "A cat's whiskers are roughly as wide as its body and help them judge spaces.",
        "Cats have a third eyelid called a nictitating membrane that protects their eyes."
    ]
    
    # 测试查询
    test_queries = [
        "How fast can cats run?",
        "How much do cats sleep?",
        "What sounds can cats make?",
        "How many toes do cats have?"
    ]
    
    # 要测试的 chunk sizes
    chunk_sizes_to_test = [100, 150, 200, 250, 300, 400]
    
    optimizer = ChunkSizeOptimizer()
    
    # 分析 chunk sizes
    analysis = optimizer.analyze_chunk_sizes(cat_facts, chunk_sizes_to_test)
    
    # 测试检索质量
    retrieval = optimizer.test_retrieval_quality(cat_facts, chunk_sizes_to_test, test_queries)
    
    # 获取推荐
    recommendation = optimizer.recommend_chunk_size(analysis, retrieval)
    
    return recommendation

if __name__ == "__main__":
    recommendation = test_cat_facts_chunking()
    
    print(f"\n✅ 建议将 config.py 中的 CHUNK_SIZE 更新为: {recommendation['recommended_size']}")