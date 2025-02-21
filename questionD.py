import itertools
from collections import defaultdict

def read_documents(filename):
    documents = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # If the line is empty I skip it
                documents.append(line)
    return documents

def compute_cooccurrences(documents):

    co_occurrence = defaultdict(int)
    
    for doc in documents:
        # Convert the document into a set of words
        words = set(doc.split())
        
        for word1, word2 in itertools.combinations(sorted(words), 2):
            co_occurrence[(word1, word2)] += 1  
    
    return co_occurrence

def main():
    # 1. Read documents from cacm.txt
    documents = read_documents("cacm.txt")
    
    # 2. Compute co-occurrences
    co_occurrence_dict = compute_cooccurrences(documents)
    
    # 3. Rank all pairs by co-occurrence counts in descending order
    sorted_pairs = sorted(co_occurrence_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 4. Print the top 10 pairs with their co-occurrence counts
    print("Top 10 most frequently co-occurring word pairs:")
    for i in range(10):
        (word1, word2), count = sorted_pairs[i]
        print(f"{word1}, {word2} => {count}")

if __name__ == "__main__":
    main()
