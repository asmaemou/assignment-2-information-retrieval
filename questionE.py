import itertools
import math
from collections import defaultdict

def main():
    # 1. I will start by reading the CACM file and gather counts
    # N represents the total number of documents
    N = 0
    # doc_freq represent how many documents contain `word`
    doc_freq = defaultdict(int)
    # cooc_freq[(word1,word2)] represent the number of documents containing both word1 and word2
    cooc_freq = defaultdict(int)

    with open("cacm.txt", "r") as f:
        for line in f:
            N += 1
            words_in_doc = set(line.strip().split())
            # increment doc freq whenever I found a word
            for w in words_in_doc:
                doc_freq[w] += 1
            # update co-occ freq
            for word1, word2 in itertools.combinations(words_in_doc, 2):
                pair_key = tuple(sorted([word1, word2]))
                cooc_freq[pair_key] += 1
    # 2. The second step involve to compute mutual information with smoothing
    # I will store each pair's MI in a dictionary:
    mi_dict = {}

    for (word1, word2), nab in cooc_freq.items():
        # Document A will contain word1, document B will contain word2
        na = doc_freq[word1]
        nb = doc_freq[word2]

        # Now I will apply smoothing: 
        # I will write this probability in my code: p(X=1, Y=1) = ( nab + 0.25 ) / ( N + 1 )
        p11 = (nab + 0.25) / (N + 1)
        # I will write this probability in my code: p(X=1, Y=0) = ( (na - nab) + 0.25 ) / (N + 1)
        p10 = ((na - nab) + 0.25) / (N + 1)
        # I will write this probability in my code: p(X=0, Y=1) = ( (nb - nab) + 0.25 ) / (N + 1)
        p01 = ((nb - nab) + 0.25) / (N + 1)
        # I will write this probability in my code: p(X=0, Y=0) = ( (N - na - nb + nab) + 0.25 ) / (N + 1)
        p00 = ((N - na - nb + nab) + 0.25) / (N + 1)

        # Then apply the marginals
        p1_ = p10 + p11  # for p(X=1)=p(X=1, Y=0)+p(X=1, Y=1)
        p0_ = p00 + p01  # for p(X=0)=p(X=0, Y=0)+p(X=0, Y=1)
        p_1 = p01 + p11  # for p(Y=1)=p(X=0, Y=1)+p(X=1, Y=1)
        p_0 = p00 + p10  # for p(Y=0)=p(X=0, Y=0)+p(X=1, Y=0)

        # the formula of mutual information is : sum p(x,y)*log2[ p(x,y)/(p(x)*p(y)) ]
        # This function safe_mi_term is a helper to handle the case if the numerator is 0:
        def safe_mi_term(p_xy, p_x, p_y):
            if p_xy <= 0:
                return 0
            ratio = p_xy / (p_x * p_y)
            return p_xy * math.log2(ratio)
        #I then coputed the mutual information value for each word pair
        mitual_information_value = 0.0 # I first initialize mutual_information to 0 to ensure that when I sum the contributions from each probability term, the calculation starts from a defined value
        mitual_information_value += safe_mi_term(p11, p1_, p_1)
        mitual_information_value += safe_mi_term(p10, p1_, p_0)
        mitual_information_value += safe_mi_term(p01, p0_, p_1)
        mitual_information_value += safe_mi_term(p00, p0_, p_0)

        mi_dict[(word1, word2)] = mitual_information_value

    # 3. After that I will sort all pairs by MI, in descending order
    # This will creates a list of word1 and word2 sorted by MI in descending order
    sorted_by_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    # I will then print top 10 pairs by MI
    print("=== Top 10 word pairs by Mutual Information ===")
    for pair, val in sorted_by_mi[:10]:
        print(f"{pair}: MI = {val:.5f}")

    # 4. I will further compare to top 10 by co-occurrence and show top 10
    sorted_by_cooc = sorted(cooc_freq.items(), key=lambda x: x[1], reverse=True)
    print("\n=== Top 10 word pairs by Co-occurrence Count ===")
    for pair, count in sorted_by_cooc[:10]:
        print(f"{pair}: Co-occurrences = {count}")

    # 5. I will find the top 5 words with highest MI with "programming", after that I will filter all pairs that involve "programming"
    programming_mi = []
    for (word1, word2), val in mi_dict.items():
        if "programming" in (word1, word2):
            # find the other word with programming
            other_word = word2 if word1 == "programming" else word1
            programming_mi.append((other_word, val))

    # I will sort and get top 5
    programming_mi.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Top 5 words with highest MI vs. 'programming' ===")
    for w, val in programming_mi[:5]:
        print(f"{w}: MI = {val:.5f}")

if __name__ == "__main__":
    main()
