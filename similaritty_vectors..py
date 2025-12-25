import os
import time
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import logging
import nltk
# Configure logging to suppress verbose gensim output
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Suppress INFO logs from gensim to keep output clean for the user
logging.getLogger("gensim").setLevel(logging.WARNING)




def setup_nltk():
    """Ensure necessary NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')

def load_data(data_dir):
    """Load and tokenize data from text files in the directory."""
    documents = []
    print(f"Loading data from {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return []

    file_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Tokenize into sentences, then words
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        # Simple preprocessing: lowercase and tokenization
                        tokens = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
                        if tokens:
                            documents.append(tokens)
                file_count += 1
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    print(f"Loaded {len(documents)} sentences from {file_count} files.")
    return documents

def format_time(seconds):
    """Format time in seconds or milliseconds."""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.4f} s"

def train_and_evaluate(documents, architecture, target_word, epochs=30):
    """Train Word2Vec model and evaluate."""
    sg_param = 1 if architecture == "Skip-gram" else 0
    print(f"\n{'='*60}")
    print(f"Training Model: {architecture}")
    print(f"{'='*60}")
    
    start_time = time.time()
    # Initialize and train the model
    # vector_size=100, window=5, min_count=2 are standard baselines
    model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4, sg=sg_param, epochs=epochs)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Training Time: {format_time(duration)}")
    
    try:
        similar_words = model.wv.most_similar(target_word.lower(), topn=10)
        print(f"\nTop 10 words similar to '{target_word}':")
        print(f"{'-'*40}")
        print(f"{'Rank':<5} | {'Word':<20} | {'Similarity':<10}")
        print(f"{'-'*40}")
        for rank, (word, similarity) in enumerate(similar_words, 1):
            print(f"{rank:<5} | {word:<20} | {similarity:.4f}")
        print(f"{'-'*40}")
    except KeyError:
        print(f"Example word '{target_word}' not found in vocabulary.")
        
    return duration

def main():
    setup_nltk()
    
    data_directory = os.path.join(os.getcwd(), "Data")
    documents = load_data(data_directory)
    
    if not documents:
        print("No data loaded. Exiting.")
        return

    target_word = "Trump" # As requested
    
    # Model A: CBOW (sg=0)
    cbow_time = train_and_evaluate(documents, "CBOW", target_word)
    
    # Model B: Skip-gram (sg=1)
    skipgram_time = train_and_evaluate(documents, "Skip-gram", target_word)
    
    # Comparison
    print(f"\n\n{'='*60}")
    print("ARCHITECHTURE BATTLE: CBOW vs. Skip-gram")
    print(f"{'='*60}")
    print(f"{'Metric':<25} | {'CBOW':<15} | {'Skip-gram':<15}")
    print(f"{'-'*60}")
    print(f"{'Training Time':<25} | {format_time(cbow_time):<15} | {format_time(skipgram_time):<15}")
    print(f"{'-'*60}")
    
    print("\nObservation:")
    if cbow_time < skipgram_time:
        factor = skipgram_time / cbow_time if cbow_time > 0 else 0
        print(f"-> CBOW was faster ({factor:.2f}x).")
    else:
        print("-> Skip-gram was faster (or equal).")
        
    print("\nNext Steps:")
    print("1. Compare quality qualitatively (see tables above).")
    print("2. Try larger datasets for more distinct performance differences.")
    print(f"{'='*60}\n")
    print("End of program. Have a nice day!")

if __name__ == "__main__":
    main()
