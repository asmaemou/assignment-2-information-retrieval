#!/usr/bin/env python3
import itertools
from collections import defaultdict

def main():
    # Dictionary to count co-occurrence of word pairs
    cooc_counts = defaultdict(int)

    # Read the CACM file (each line is a document)
    with open('cacm.txt', 'r') as f:
        for line in f:
            # Get unique words from the document
            words = set(line.strip().split())
            
            # Generate all unordered pairs of words in the document
            for A, B in itertools.combinations(words, 2):
                # Normalize the order of the pair to avoid duplication (use (min, max))
                pair_key = (min(A, B), max(A, B))
                cooc_counts[pair_key] += 1

    # Sort cooc_counts by value (count) in descending order and get the top 10
    top_10 = sorted(cooc_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print the top 10 pairs and their counts
    for pair, count in top_10:
        print(f"{pair}: {count}")

if __name__ == "__main__":
    main()
