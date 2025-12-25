import os
import sys
import time
import logging
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
logging.getLogger("gensim").setLevel(logging.WARNING)


# -------------------------
# NLTK setup (STRICT)
# -------------------------
def setup_nltk():
    """
    Ensure required NLTK resources exist.
    If they cannot be prepared, fail hard.
    """
    required_resources = [
        "tokenizers/punkt"
    ]

    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"NLTK resource missing: {resource}. Downloading...")
            nltk.download("punkt")

            # Verify again
            try:
                nltk.data.find(resource)
            except LookupError:
                print(f"FATAL: Failed to prepare NLTK resource: {resource}")
                sys.exit(1)  # <-- CI FAILS HERE


# -------------------------
# Data loading (STRICT)
# -------------------------
def load_data(data_dir):
    documents = []
    print(f"Loading data from {data_dir}...")

    if not os.path.exists(data_dir):
        print(f"FATAL: Data directory not found: {data_dir}")
        sys.exit(1)  # <-- CI FAILS HERE

    file_count = 0
    error_count = 0

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    sentences = sent_tokenize(text)

                    for sentence in sentences:
                        tokens = [
                            word.lower()
                            for word in word_tokenize(sentence)
                            if word.isalnum()
                        ]
                        if tokens:
                            documents.append(tokens)

                file_count += 1

            except Exception as e:
                error_count += 1
                print(f"Error reading {filename}: {e}")

    print(f"Processed {file_count} files with {error_count} errors.")
    print(f"Loaded {len(documents)} sentences.")

    if not documents:
        print("FATAL: No usable data loaded.")
        sys.exit(1)  # <-- CI FAILS HERE

    return documents


# -------------------------
# Utility
# -------------------------
def format_time(seconds):
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.4f} s"


# -------------------------
# Training
# -------------------------
def train_and_evaluate(documents, architecture, target_word, epochs=30):
    sg_param = 1 if architecture == "Skip-gram" else 0

    print(f"\n{'='*60}")
    print(f"Training Model: {architecture}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        model = Word2Vec(
            sentences=documents,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=sg_param,
            epochs=epochs
        )
    except Exception as e:
        print(f"FATAL: Word2Vec training failed: {e}")
        sys.exit(1)  # <-- CI FAILS HERE

    duration = time.time() - start_time
    print(f"Training Time: {format_time(duration)}")

    try:
        similar_words = model.wv.most_similar(target_word.lower(), topn=10)
        print(f"\nTop words similar to '{target_word}':")
        for rank, (word, similarity) in enumerate(similar_words, 1):
            print(f"{rank:>2}. {word:<20} {similarity:.4f}")
    except KeyError:
        print(f"Warning: '{target_word}' not in vocabulary.")

    return duration


# -------------------------
# Main (THE JUDGE)
# -------------------------
def main():
    setup_nltk()

    data_directory = os.path.join(os.getcwd(), "Data")
    documents = load_data(data_directory)

    target_word = "Trump"

    cbow_time = train_and_evaluate(documents, "CBOW", target_word)
    skipgram_time = train_and_evaluate(documents, "Skip-gram", target_word)

    print(f"\n{'='*60}")
    print("ARCHITECTURE BATTLE: CBOW vs. Skip-gram")
    print(f"{'='*60}")
    print(f"CBOW Time     : {format_time(cbow_time)}")
    print(f"Skip-gram Time: {format_time(skipgram_time)}")

    print("\nRun completed successfully.")


if __name__ == "__main__":
    main()
