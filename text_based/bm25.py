import math
import heapq

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

from text_based.spimi import tokenize


class BM25:
    def __init__(self, spimi_index, k1=DEFAULT_K1, b=DEFAULT_B):
        self.index = spimi_index
        self.avg_doc_length = 0
        self.doc_frequencies = {}
        self.ready = False
        self.k1 = k1
        self.b = b

    def build_stats(self):
        self.index.load()

        self.doc_frequencies = {}
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

        for term, df in self.index.iter_postings_counts():
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

        terms = []
        seen = set()
        for word in tokenize(query):
            if word not in seen:
                seen.add(word)
                terms.append(word)

        scores = {}

        for term in terms:
            postings = self.index.get_postings(term)
            if not postings:
                continue

            total_doc_frequency = self.doc_frequencies[term]
            # IDF de BM25.  Se usa log(1 + ...) para que nunca sea negativo,
            # incluso cuando el termino aparece en mas de la mitad del corpus.
            idf = math.log(
                1.0
                + (amount_of_documents - total_doc_frequency + 0.5)
                / (total_doc_frequency + 0.5)
            )

            for doc_id, frequency in postings:
                doc_length = self.index.doc_lengths[doc_id]
                norm = 1 - self.b + self.b * doc_length / self.avg_doc_length
                tf = (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * norm)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

        return heapq.nlargest(
            top_k,
            scores.items(),
            key=lambda item: item[1]
        )
