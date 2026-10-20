"""Shared loader for the pipeline the Master notebook saves in Section 9.

Every Lab 2 agent tool reads its inputs from lab2_workspace/ on disk,
never from a running notebook's variables. That is what lets
DM2026-Lab2-AgentPipeline.ipynb start in a fresh kernel and still
evaluate exactly the vectorizer, embeddings, and models the Master
built. The files are loaded once per session, on first use, and cached
in session.artifacts["lab2_workspace"].

This module is imported by the tool files in lab2_tools/experiments/
and lab2_tools/inspection/; it defines no tools itself, so
agent_pipeline.workspace_loader never scans it (it only scans those two
directories).
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

WORKSPACE_DIR = "lab2_workspace"
REPRESENTATIONS = ("tfidf", "word2vec", "doc2vec", "fasttext", "minilm")
RETRIEVAL_REPRESENTATIONS = ("tfidf", "word2vec", "minilm")
MAX_DOCUMENTS_PER_CALL = 10

# How the Master notebook labels each representation in its results table,
# so a run here can be matched against the Master's own number.
MASTER_REPRESENTATION_NAMES = {
    "tfidf": "TF-IDF",
    "word2vec": "Word2Vec-mean",
    "doc2vec": "Doc2Vec",
    "fasttext": "FastText-mean",
    "minilm": "MiniLM",
}


def clean_text(s: str) -> str:
    """Same normalization as the Master notebook's clean_text (Section 2)."""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_matrix(filename):
    path = os.path.join(WORKSPACE_DIR, filename)
    if filename.endswith(".npz"):
        return scipy.sparse.load_npz(path)
    return np.load(path)


def load_workspace(session) -> dict:
    """Loads lab2_workspace/ into session.artifacts on first call, then
    returns the cached copy. Raises FileNotFoundError with a plain
    instruction if the Master notebook hasn't saved the pipeline yet."""
    cached = session.artifacts.get("lab2_workspace")
    if cached is not None:
        return cached

    manifest_path = os.path.join(WORKSPACE_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "lab2_workspace/manifest.json not found. Run DM2026-Lab2-Master.ipynb "
            "through Section 9 (Saving the Fitted Pipeline) first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    workspace = {
        "manifest": manifest,
        "labels": manifest["labels"],
        "train": pd.read_csv(os.path.join(WORKSPACE_DIR, manifest["frames"]["train"])),
        "test": pd.read_csv(os.path.join(WORKSPACE_DIR, manifest["frames"]["test"])),
        "matrices": {},
        "tfidf_vectorizer": joblib.load(os.path.join(WORKSPACE_DIR, manifest["tfidf_vectorizer"])),
        "word2vec": KeyedVectors.load(os.path.join(WORKSPACE_DIR, manifest["word2vec_vectors"])),
        "master_results": pd.read_csv(os.path.join(WORKSPACE_DIR, manifest["master_results"])),
        "minilm_model": None,  # loaded on the first MiniLM query, not before
    }
    for name, spec in manifest["representations"].items():
        workspace["matrices"][name] = (_load_matrix(spec["train"]), _load_matrix(spec["test"]))

    session.artifacts["lab2_workspace"] = workspace
    return workspace


def workspace_error(exc) -> dict:
    """The structured error every tool returns when the workspace is missing."""
    return {"error": str(exc), "hint": "Tell the student which notebook section to run; do not retry."}


def embed_query(workspace, representation: str, query: str):
    """Embeds one query string the same way the Master's retrieve()
    function does for that representation. Returns a 2-D array (1 row)."""
    if representation == "tfidf":
        return workspace["tfidf_vectorizer"].transform([clean_text(query)])
    if representation == "word2vec":
        vectors = workspace["word2vec"]
        known = [vectors[t] for t in word_tokenize(clean_text(query)) if t in vectors]
        mean = np.mean(known, axis=0) if known else np.zeros(vectors.vector_size, dtype=np.float32)
        return mean.reshape(1, -1)
    if representation == "minilm":
        if workspace["minilm_model"] is None:
            from sentence_transformers import SentenceTransformer

            workspace["minilm_model"] = SentenceTransformer(workspace["manifest"]["minilm_model"])
        return workspace["minilm_model"].encode([query])
    raise ValueError(f"Unknown retrieval representation '{representation}'.")


def rank_training_documents(workspace, representation: str, query: str, k: int):
    """Returns (row_indices, similarities) of the k training tweets most
    similar to `query`, ordered exactly as the Master's retrieve()."""
    X_train = workspace["matrices"][representation][0]
    sims = cosine_similarity(embed_query(workspace, representation, query), X_train).ravel()
    top = sims.argsort()[::-1][:k]
    return top, sims[top]


def document_rows(workspace, indices, similarities) -> list:
    """Turns retrieved training-row indices into small records for the agent."""
    train = workspace["train"]
    return [
        {
            "rank": rank,
            "doc_id": int(train["id"].iloc[i]),
            "similarity": round(float(s), 4),
            "emotion": str(train["emotion"].iloc[i]),
            "text": str(train["text"].iloc[i]),
        }
        for rank, (i, s) in enumerate(zip(indices, similarities), start=1)
    ]


def precision_at_k(workspace, representation: str, k: int):
    """Label precision@k: every test tweet is used as a query against the
    training tweets, and a retrieved tweet counts as relevant when it
    carries the query's emotion label. Same computation as the Master's
    Section 8.2. Returns (overall, per_emotion dict, n_queries)."""
    X_train, X_test = workspace["matrices"][representation]
    y_train = workspace["train"]["emotion"].to_numpy()
    y_test = workspace["test"]["emotion"].to_numpy()
    sims = cosine_similarity(X_test, X_train)
    top = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    hits = y_train[top] == y_test[:, None]
    per_emotion = {label: float(hits[y_test == label].mean()) for label in workspace["labels"]}
    return float(hits.mean()), per_emotion, int(len(y_test))


def random_precision_baseline(workspace) -> float:
    """Expected label precision of a retriever that returns random tweets."""
    train_share = workspace["train"]["emotion"].value_counts(normalize=True)
    test_share = workspace["test"]["emotion"].value_counts(normalize=True)
    return float(sum(train_share.get(label, 0) * test_share.get(label, 0) for label in workspace["labels"]))
