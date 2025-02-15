#!/usr/bin/env python3
import itertools
import math
from collections import defaultdict

def main():
    ########################################
    # 1. Read the CACM file and gather counts
    ########################################
    # N = total number of documents
    N = 0
    # doc_freq[word] = how many documents contain `word`
    doc_freq = defaultdict(int)
    # cooc_freq[(w1,w2)] = number of documents containing both w1 and w2
    cooc_freq = defaultdict(int)

    with open("cacm.txt", "r") as f:
        for line in f:
            N += 1
            words_in_doc = set(line.strip().split())
            # update doc freq
            for w in words_in_doc:
                doc_freq[w] += 1
            # update co-occ freq (unordered pairs)
            for w1, w2 in itertools.combinations(words_in_doc, 2):
                pair_key = tuple(sorted([w1, w2]))
                cooc_freq[pair_key] += 1

    ########################################
    # 2. Compute mutual information with smoothing
    ########################################
    # We'll store each pair's MI in a dictionary:
    mi_dict = {}

    for (w1, w2), nab in cooc_freq.items():
        # NA = doc_freq[w1], NB = doc_freq[w2]
        na = doc_freq[w1]
        nb = doc_freq[w2]

        # Smoothing: 
        # p(X=1, Y=1) = ( nab + 0.25 ) / ( N + 1 )
        p11 = (nab + 0.25) / (N + 1)
        # p(X=1, Y=0) = ( (na - nab) + 0.25 ) / (N + 1)
        p10 = ((na - nab) + 0.25) / (N + 1)
        # p(X=0, Y=1) = ( (nb - nab) + 0.25 ) / (N + 1)
        p01 = ((nb - nab) + 0.25) / (N + 1)
        # p(X=0, Y=0) = ( (N - na - nb + nab) + 0.25 ) / (N + 1)
        p00 = ((N - na - nb + nab) + 0.25) / (N + 1)

        # Marginals
        p1_ = p10 + p11  # p(X=1)
        p0_ = p00 + p01  # p(X=0)
        p_1 = p01 + p11  # p(Y=1)
        p_0 = p00 + p10  # p(Y=0)

        # Mutual information sum p(x,y)*log2[ p(x,y)/(p(x)*p(y)) ]
        # We'll define a small helper to handle 0 safely:
        def safe_mi_term(p_xy, p_x, p_y):
            if p_xy <= 0:
                return 0
            ratio = p_xy / (p_x * p_y)
            return p_xy * math.log2(ratio)

        mi_val = 0.0
        mi_val += safe_mi_term(p11, p1_, p_1)
        mi_val += safe_mi_term(p10, p1_, p_0)
        mi_val += safe_mi_term(p01, p0_, p_1)
        mi_val += safe_mi_term(p00, p0_, p_0)

        mi_dict[(w1, w2)] = mi_val

    ########################################
    # 3. Sort all pairs by MI, descending
    ########################################
    # This creates a list of ( (w1,w2), MI ) sorted by MI
    sorted_by_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    # Print top 10 pairs by MI
    print("=== Top 10 word pairs by Mutual Information ===")
    for pair, val in sorted_by_mi[:10]:
        print(f"{pair}: MI = {val:.5f}")

    ########################################
    # 4. Compare to top 10 by co-occurrence
    ########################################
    # For reference, let's also show top 10 by raw co-occurrence:
    sorted_by_cooc = sorted(cooc_freq.items(), key=lambda x: x[1], reverse=True)
    print("\n=== Top 10 word pairs by Co-occurrence Count ===")
    for pair, count in sorted_by_cooc[:10]:
        print(f"{pair}: Co-occurrences = {count}")

    ########################################
    # 5. Find the top 5 words with highest MI with "programming"
    ########################################
    # We'll filter all pairs that involve "programming"
    programming_mi = []
    for (w1, w2), val in mi_dict.items():
        if "programming" in (w1, w2):
            # figure out which is the "other" word
            other_word = w2 if w1 == "programming" else w1
            programming_mi.append((other_word, val))

    # sort and get top 5
    programming_mi.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Top 5 words with highest MI vs. 'programming' ===")
    for w, val in programming_mi[:5]:
        print(f"{w}: MI = {val:.5f}")

if __name__ == "__main__":
    main()
