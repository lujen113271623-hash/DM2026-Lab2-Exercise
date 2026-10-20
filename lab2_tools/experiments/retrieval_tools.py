"""Experiment tools for information retrieval: retrieve_tool,
compare_retrieval_tool, and evaluate_retrieval_tool.

They run the retrieval the Master notebook teaches in Section 8 over the
saved index in lab2_workspace/: the training tweets' vectors under three
representations (TF-IDF, Word2Vec, MiniLM), queried by cosine
similarity. They sit behind the request/grant gate because they compute
new results and, for retrieve_tool and compare_retrieval_tool, send
retrieved tweets to the language model.
"""

from typing import Literal

from langchain_core.tools import tool

from lab2_tools.workspace import (
    MAX_DOCUMENTS_PER_CALL,
    RETRIEVAL_REPRESENTATIONS,
    document_rows,
    load_workspace,
    precision_at_k,
    random_precision_baseline,
    rank_training_documents,
    workspace_error,
)

MAX_COMPARE_K = 5


def _emotion_counts(rows):
    counts = {}
    for row in rows:
        counts[row["emotion"]] = counts.get(row["emotion"], 0) + 1
    return counts


def make_tools(session):
    @tool
    def retrieve_tool(
        query: str,
        representation: Literal["tfidf", "word2vec", "minilm"],
        k: int = 5,
    ) -> dict:
        """Retrieve the k training tweets most similar to a query under one
        representation, ranked by cosine similarity. Returns rank, doc_id,
        similarity, emotion and text, at most 10 since tweets are sent
        here.

        Args:
            query: the search text, in plain words.
            representation: tfidf matches shared words, word2vec averaged
                self-trained vectors, minilm pre-trained meaning.
            k: how many tweets, 1 to 10.
        """
        # For the reader, not the model: this is the same computation as the
        # Master's retrieve(), so a query run here reproduces what Section 8
        # showed.

        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        if not query.strip():
            return {"error": "The query is empty."}
        k_used = max(1, min(int(k), MAX_DOCUMENTS_PER_CALL))
        indices, similarities = rank_training_documents(workspace, representation, query, k_used)
        rows = document_rows(workspace, indices, similarities)

        result_id = session.next_result_id("retrieve")
        summary = {
            "result_id": result_id,
            "query": query,
            "representation": representation,
            "k": k_used,
            "documents": rows,
            "emotion_counts": _emotion_counts(rows),
        }
        session.store_result("retrieve", {"query": query, "representation": representation, "k": k_used}, summary)
        return summary

    @tool
    def compare_retrieval_tool(query: str, k: int = 5) -> dict:
        """Run one query through all three retrievers (tfidf, word2vec,
        minilm) and return their tweets side by side plus how many
        documents each pair shares. At most 5 per retriever.

        Args:
            query: the search text, in plain words.
            k: tweets per retriever, 1 to 5.
        """
        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        if not query.strip():
            return {"error": "The query is empty."}
        k_used = max(1, min(int(k), MAX_COMPARE_K))

        rows_by_rep = {}
        for representation in RETRIEVAL_REPRESENTATIONS:
            indices, similarities = rank_training_documents(workspace, representation, query, k_used)
            rows_by_rep[representation] = document_rows(workspace, indices, similarities)

        overlap = []
        pairs = [("tfidf", "word2vec"), ("tfidf", "minilm"), ("word2vec", "minilm")]
        for a, b in pairs:
            shared = sorted({r["doc_id"] for r in rows_by_rep[a]} & {r["doc_id"] for r in rows_by_rep[b]})
            overlap.append({"pair": f"{a} vs {b}", "shared_documents": len(shared), "shared_doc_ids": shared})

        result_id = session.next_result_id("compare_retrieval")
        summary = {
            "result_id": result_id,
            "query": query,
            "k": k_used,
            "tfidf_documents": rows_by_rep["tfidf"],
            "word2vec_documents": rows_by_rep["word2vec"],
            "minilm_documents": rows_by_rep["minilm"],
            "overlap": overlap,
            "emotion_counts": {rep: _emotion_counts(rows) for rep, rows in rows_by_rep.items()},
        }
        session.store_result("compare_retrieval", {"query": query, "k": k_used}, summary)
        return summary

    @tool
    def evaluate_retrieval_tool(
        representation: Literal["tfidf", "word2vec", "minilm"],
        k: int = 5,
    ) -> dict:
        """Label precision@k for one retriever over the whole test set: each
        test tweet queries the training tweets, and a retrieved tweet
        counts when it carries the query's emotion. Returns overall and
        per-emotion precision plus the random baseline. Sends no tweets.

        Args:
            representation: tfidf, word2vec, or minilm.
            k: retrieved tweets scored per query, 1 to 50.
        """
        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        if not 1 <= k <= 50:
            return {"error": "k must be between 1 and 50."}
        overall, per_emotion, n_queries = precision_at_k(workspace, representation, int(k))

        result_id = session.next_result_id("evaluate_retrieval")
        summary = {
            "result_id": result_id,
            "representation": representation,
            "k": int(k),
            "n_queries": n_queries,
            "precision_at_k": round(overall, 4),
            "random_baseline": round(random_precision_baseline(workspace), 4),
            "per_emotion": [
                {"emotion": label, "precision_at_k": round(value, 4)} for label, value in per_emotion.items()
            ],
        }
        session.store_result("evaluate_retrieval", {"representation": representation, "k": int(k)}, summary)
        return summary

    return [retrieve_tool, compare_retrieval_tool, evaluate_retrieval_tool]
