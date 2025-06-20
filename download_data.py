# download_data.py
import requests

def download_cat_facts():
    """Download the sample cat facts dataset"""
    url = "https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open('data/cat-facts.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print("✅ Dataset downloaded successfully")
        
        # Display sample data
        with open('data/cat-facts.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"📊 Total facts: {len(lines)}")
            print("📝 Sample facts:")
            for i, fact in enumerate(lines[:3]):
                print(f"  {i+1}. {fact.strip()}")
                
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Creating sample dataset manually...")
        create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset if download fails"""
    sample_facts = [
        "Cats can travel at a top speed of approximately 31 mph (49 km) over a short distance.",
        "A cat's hearing is better than a dog's. Cats can hear high-frequency sounds up to 64,000 Hz.",
        "Cats have a special scent organ called the Jacobson's organ in the roof of their mouths.",
        "A group of cats is called a clowder, and a group of kittens is called a kindle.",
        "Cats spend 70% of their lives sleeping, which is 13-16 hours a day.",
        "A cat's purr vibrates at a frequency of 25-50 Hz, which can lower blood pressure and promote healing.",
        "Cats have five toes on their front paws but only four on their back paws.",
        "The oldest known pet cat existed 9,500 years ago in Cyprus.",
        "Cats can make over 100 different vocal sounds, while dogs can only make 10.",
        "A cat's nose print is unique, much like a human's fingerprint."
    ]
    
    with open('data/cat-facts.txt', 'w', encoding='utf-8') as f:
        for fact in sample_facts:
            f.write(fact + '\n')
    
    print("✅ Sample dataset created")

if __name__ == "__main__":
    download_cat_facts()