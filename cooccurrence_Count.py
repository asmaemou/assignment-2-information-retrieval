# # defaultdict: This is used to create a hash table (dictionary) to count co-occurrences for each word pair. It initializes values to 0 by default.
from collections import defaultdict
import itertools
# # itertools.combinations: This generates all possible unique combinations of two words from a set of words in a document.
pair_counts = defaultdict(int)



with open('cacm.txt', 'r') as f:
    for line in f:
        # Remove leading/trailing whitespace and split into words
        # Using set() ensures that repeated words in one document are counted only once
        words = set(line.strip().split())
        # Generate all unique unordered pairs of words (sorted to avoid duplicate pairs)
        for pair in itertools.combinations(sorted(words), 2):
            pair_counts[pair] += 1

# Sort the counts in descending order and extract the top 10 counts
top_10_counts = sorted(pair_counts.values(), reverse=True)[:10]

# Print the top 10 counts, one count per line
for count in top_10_counts:
    print(count)
