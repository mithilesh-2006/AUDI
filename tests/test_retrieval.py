"""
Unit tests for Local Evidence Retrieval Engine.
Verifies BM25 passage matching, threshold gating, no-evidence handling, and dynamic indexing.
"""

import unittest
from src.retrieval import LocalCorpusRetriever, RetrievedPassage


class TestLocalCorpusRetriever(unittest.TestCase):

    def setUp(self):
        # In-memory retriever with controlled documents
        self.retriever = LocalCorpusRetriever(corpus_path="data/corpus/test_corpus.json")
        # Clear any existing and add test docs
        self.retriever.documents.clear()
        self.retriever.passages.clear()
        
        self.retriever.add_document(
            doc_id="test_everest",
            title="Mount Everest Elevation",
            source="Survey 2020",
            text="Mount Everest official elevation is 8,848.86 meters above sea level.",
            save=False
        )
        self.retriever.add_document(
            doc_id="test_moon",
            title="Apollo 11 Mission",
            source="NASA 1969",
            text="Apollo 11 landed on the Moon on July 20, 1969 with Neil Armstrong.",
            save=False
        )

    def test_matching_query_retrieval(self):
        res = self.retriever.retrieve("What is the elevation of Mount Everest?")
        self.assertTrue(res.has_sufficient_evidence)
        self.assertGreater(len(res.passages), 0)
        top = res.passages[0]
        self.assertEqual(top.doc_id, "test_everest")
        self.assertIn("8,848.86", top.text)
        self.assertIn("Survey 2020", top.source_url_or_ref)

    def test_nonmatching_query_retrieval(self):
        res = self.retriever.retrieve("Quantum chromodynamics gluon entanglement quark plasma")
        self.assertFalse(res.has_sufficient_evidence)
        self.assertIn("No matching evidence", res.message)

    def test_empty_query_handling(self):
        res = self.retriever.retrieve("")
        self.assertFalse(res.has_sufficient_evidence)
        self.assertEqual(len(res.passages), 0)

    def test_dynamic_document_addition(self):
        self.retriever.add_document(
            doc_id="doc_new_einstein",
            title="Einstein Theory of General Relativity",
            source="Physics Archives 1915",
            text="Albert Einstein published the general theory of relativity in 1915.",
            save=False
        )
        res = self.retriever.retrieve("When was general relativity published?")
        self.assertTrue(res.has_sufficient_evidence)
        self.assertEqual(res.passages[0].doc_id, "doc_new_einstein")


if __name__ == "__main__":
    unittest.main()
