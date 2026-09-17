"""Exercise the imported Query flow with explicit model and corpus boundaries."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Telco-RAG_api"))
sys.path.insert(0, str(ROOT / "extensions/rag-lab"))
from src.corpus import StoreCorpus
from src.query import Query
from raglab.ingest import ingest_file
from raglab.store import SQLiteStore



class QueryFlowTests(unittest.TestCase):
    def test_two_independent_corpora_use_original_candidate_stage(self):
        generate = Mock(return_value="timer")
        with tempfile.TemporaryDirectory() as folder, SQLiteStore(":memory:") as store:
            path = Path(folder) / "manual.txt"
            for corpus, sentence in [("telecom", "timer lasts 30 seconds"), ("equipment", "timer lasts 5 seconds")]:
                path.write_text(sentence)
                store.replace_document(corpus, "manual", ingest_file(path, corpus_id=corpus, document_id="manual"))
                query = Query("timer", [], corpus=StoreCorpus(store, corpus_ids=[corpus]), completion=generate)
                query.get_3GPP_context(validate_flag=False)
                self.assertEqual(len(query.context), 1)
                self.assertIn(sentence, query.context[0])
                self.assertIn(corpus + "/", query.context[0])
                self.assertEqual(len(query.context_source), 1)
        self.assertEqual(generate.call_count, 2)

    def test_empty_corpus_does_not_generate_candidates(self):
        generate = Mock()
        corpus = Mock()
        corpus.search.return_value = []
        query = Query("question", [], corpus=corpus, completion=generate)
        query.get_3GPP_context()
        generate.assert_not_called()
        self.assertEqual(query.context, [])

    def test_newly_routed_series_actually_loads_documents(self):
        load = Mock(side_effect=[[{"source": "23.001"}], [{"source": "38.001"}]])
        query = Query.__new__(Query)
        query.corpus = None
        selections = iter([[23], [38]])
        query.predict_wg = lambda: setattr(query, "wg", next(selections))
        query.candidate_answers = Mock()
        query.get_question_context_faiss = Mock()
        import types
        modules = {}
        for name, symbol, value in [('src.input', 'get_documents', load), ('src.chunking', 'chunk_doc', lambda d: [d]), ('src.embeddings', 'get_embeddings', lambda groups: groups)]:
            modules[name] = types.ModuleType(name)
            setattr(modules[name], symbol, value)
        with patch.dict(sys.modules, modules):
            query.get_3GPP_context(validate_flag=False)
        self.assertEqual(load.call_args_list[1].args[0], [38])
        last_batch = query.get_question_context_faiss.call_args.kwargs["batch"]
        self.assertIn([[{"source": "38.001"}]], last_batch)


if __name__ == "__main__":
    unittest.main()

class RetrievalRegressionTests(unittest.TestCase):
    def load_functions(self, **dependencies):
        tree = ast.parse((ROOT / "Telco-RAG_api/src/retrieval.py").read_text())
        namespace = dict(dependencies)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), "retrieval.py", "exec"), namespace)
        return namespace

    def test_string_context_is_not_split_into_characters(self):
        embedding = Mock()
        embedding.return_value.data = [Mock(embedding=[1.0, 2.0])]
        np = Mock()
        np.array.side_effect = lambda values, dtype: values
        functions = self.load_functions(embedding=embedding, np=np)
        functions['get_query_embedding_OpenAILarge']("question", "candidate")
        embedding.assert_called_once_with("question\ncandidate")

    def test_faiss_missing_neighbor_is_not_returned_as_evidence(self):
        functions = self.load_functions()
        functions['get_query_embedding_OpenAILarge'] = lambda *args: [1.0]
        functions['search_faiss_index'] = lambda *args: ([[0, -1]], [[1.0, 0.0]])
        found = functions['find_nearest_neighbors_faiss']("q", None, {0: "real"}, 2, {0: "source"}, {0: [1.0]})
        self.assertEqual(found, [(0, "real", "source", [1.0])])
