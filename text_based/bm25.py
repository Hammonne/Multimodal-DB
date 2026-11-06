import math
import heapq

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

from text_based.spimi import tokenize

class BM25:
    def __init__(self, spimi_index, k1 = DEFAULT_K1, b = DEFAULT_B):
        self.index = spimi_index
        self.avg_doc_length = 0
        self.doc_frequencies = {}
        self.ready = False
        self.k1 = k1
        self.b = b

    def build_stats(self):
        self.index.load()

        doc_length_sum = 0
        amount_of_documents = len(self.index.doc_lengths)
        if amount_of_documents == 0:
            self.avg_doc_length = 0.0
            self.ready = True
            return
        for doc_id in self.index.doc_lengths:
            doc_length_sum += self.index.doc_lengths[doc_id]
        
        self.avg_doc_length = doc_length_sum / amount_of_documents
        if self.avg_doc_length == 0:
            self.ready = True
            return

        for term in self.index.term_offsets:
            postings = self.index.get_postings(term)
            df = len(postings)
            if df == 0:
                continue
            self.doc_frequencies[term] = df

        self.ready = True

    def search(self, query, top_k):
        if not self.ready:
            self.build_stats()

        amount_of_documents = len(self.index.doc_lengths)
        if amount_of_documents == 0 or self.avg_doc_length == 0:
            return []

        if top_k <= 0:
            return []

        query_tf = {}

        for word in tokenize(query):
            query_tf[word] = query_tf.get(word, 0) + 1

        scoresdict = {}

        for term in query_tf:
            postings = self.index.get_postings(term)
            if not postings:
                continue
            total_doc_frequency = self.doc_frequencies[term]
            IDF = math.log((amount_of_documents - total_doc_frequency + 0.5) / (total_doc_frequency + 0.5))
            for doc_id, doc_frequency in postings:
                doc_length = self.index.doc_lengths[doc_id]
                thirdterm = (1 - self.b + self.b * doc_length / self.avg_doc_length)
                TF = doc_frequency / (doc_frequency + self.k1 + thirdterm)
                scoresdict[doc_id] = scoresdict.get(doc_id, 0.0) + TF*IDF
        scores = list(scoresdict.items())

        return heapq.nlargest(
            top_k,
            scores,
            key=lambda item: item[1]
        )

