import math
import heapq

from text_based.spimi import tokenize

class TFIDFCosine:
    def __init__(self, spimi_index):
        self.index = spimi_index
        self.doc_norms = {}
        self.ready = False

    def build_stats(self):
        self.index.load()

        for doc_id in self.index.doc_lengths:
            self.doc_norms[doc_id] = 0.0

        for term in self.index.term_offsets:
            postings = self.index.get_postings(term)

            df = len(postings)

            if df == 0:
                continue

            for doc_id, frequency in postings:
                tf = 1 + math.log10(frequency)
                weight = tf / df

                self.doc_norms[doc_id] += weight ** 2

        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(
                self.doc_norms[doc_id]
            )

        self.ready = True

    def search(self, query, top_k):
        if not self.ready:
            self.build_stats()

        if top_k <= 0:
            return []

        query_tf = {}

        for word in tokenize(query):
            query_tf[word] = query_tf.get(word, 0) + 1

        query_weights = {}
        query_norm_squared = 0.0

        for term, frequency in query_tf.items():
            postings = self.index.get_postings(term)

            if not postings:
                continue

            df = len(postings)
            tf = 1 + math.log10(frequency)
            weight = tf / df

            query_weights[term] = weight
            query_norm_squared += weight ** 2

        query_norm = math.sqrt(query_norm_squared)

        if query_norm == 0:
            return []

        dot_products = {}

        for term, query_weight in query_weights.items():
            postings = self.index.get_postings(term)

            df = len(postings)

            for doc_id, frequency in postings:
                tf = 1 + math.log10(frequency)
                doc_weight = tf / df

                contribution = query_weight * doc_weight

                dot_products[doc_id] = (
                    dot_products.get(doc_id, 0.0) + contribution
                )

        scores = []

        for doc_id, dot_product in dot_products.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)

            if doc_norm == 0:
                continue

            score = dot_product / (query_norm * doc_norm)
            scores.append((doc_id, score))

        return heapq.nlargest(
            top_k,
            scores,
            key=lambda item: item[1]
        )

