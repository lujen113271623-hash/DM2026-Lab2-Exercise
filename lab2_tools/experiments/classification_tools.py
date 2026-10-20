"""Experiment tools for classification: run_classifier_tool and
show_misclassified_tool.

These sit behind the request/grant gate (agent_pipeline/tool_access.py):
the agent must ask the student before its first use of each one, because
run_classifier_tool computes a new result and show_misclassified_tool
sends actual tweets to the language model. Both read only from
lab2_workspace/, the files DM2026-Lab2-Master.ipynb saves in Section 9.
"""

import time
from typing import List, Literal, Optional

from langchain_core.tools import tool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lab2_tools.workspace import (
    MASTER_REPRESENTATION_NAMES,
    MAX_DOCUMENTS_PER_CALL,
    load_workspace,
    workspace_error,
)

SEED = 42


def _build_model(classifier, is_sparse, k, C, hidden_layer_sizes):
    """Builds the same model families, with the same settings, as the
    Master notebook's Sections 6 and 7."""
    if classifier == "knn":
        return KNeighborsClassifier(n_neighbors=k), {"k": k}
    if classifier == "linear_svm":
        return LinearSVC(C=C, random_state=SEED), {"C": C}
    if classifier == "naive_bayes":
        # MultinomialNB models non-negative counts; dense embeddings need GaussianNB.
        return (MultinomialNB(), {"variant": "multinomial"}) if is_sparse else (GaussianNB(), {"variant": "gaussian"})
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
    )
    params = {"hidden_layer_sizes": list(hidden_layer_sizes)}
    if is_sparse:
        return mlp, params
    return Pipeline([("scale", StandardScaler()), ("mlp", mlp)]), params


def _master_label(classifier, params):
    if classifier == "knn":
        return f"KNN (k={params['k']})"
    if classifier == "linear_svm":
        return "Linear SVM" if params["C"] == 1.0 else None
    if classifier == "naive_bayes":
        return "Naive Bayes"
    sizes = params["hidden_layer_sizes"]
    return f"MLP ({', '.join(str(s) for s in sizes)})" if len(sizes) > 1 else f"MLP ({sizes[0]},)"


def make_tools(session):
    @tool
    def run_classifier_tool(
        representation: Literal["tfidf", "word2vec", "doc2vec", "fasttext", "minilm"],
        classifier: Literal["knn", "linear_svm", "naive_bayes", "mlp"],
        k: int = 5,
        C: float = 1.0,
        hidden_layer_sizes: List[int] = [128, 64],
    ) -> dict:
        """Train one classifier on one saved representation and evaluate it on
        the held-out test tweets. Returns accuracy, macro F1, per-class F1,
        and a result_id. Includes master_reference_macro_f1 when the
        Master's table holds the same configuration. Only the parameter
        belonging to the chosen classifier is used.

        Args:
            representation: tfidf is sparse, the other four are dense.
            classifier: knn, linear_svm, naive_bayes, or mlp.
            k: neighbors for knn, 1 to 200.
            C: regularization for linear_svm, above 0; smaller is stronger.
            hidden_layer_sizes: mlp layers, 1 to 3 of 1 to 512 neurons,
                e.g. [128, 64].
        """
        # For the reader, not the model: naive_bayes is MultinomialNB on
        # tfidf and GaussianNB on the dense embeddings, since multinomial
        # cannot take negative features; mlp stops early on a 15% validation
        # slice. The result_id this returns is what inspect_result_tool,
        # compare_results_tool and show_misclassified_tool take, and
        # master_reference_macro_f1 is what lets a run be checked against
        # the Master's own number.

        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        if classifier == "knn" and not 1 <= k <= 200:
            return {"error": "k must be between 1 and 200."}
        if classifier == "linear_svm" and not C > 0:
            return {"error": "C must be greater than 0."}
        if classifier == "mlp" and not (1 <= len(hidden_layer_sizes) <= 3 and all(1 <= s <= 512 for s in hidden_layer_sizes)):
            return {"error": "hidden_layer_sizes must be 1 to 3 layers of 1 to 512 neurons each."}

        X_train, X_test = workspace["matrices"][representation]
        y_train = workspace["train"]["emotion"].to_numpy()
        y_test = workspace["test"]["emotion"].to_numpy()
        is_sparse = workspace["manifest"]["representations"][representation]["kind"] == "sparse"
        model, params = _build_model(classifier, is_sparse, k, float(C), hidden_layer_sizes)

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        seconds = time.time() - start

        labels = workspace["labels"]
        per_class = f1_score(y_test, y_pred, labels=labels, average=None)
        master = workspace["master_results"]
        master_label = _master_label(classifier, params)
        match = master[
            (master["representation"] == MASTER_REPRESENTATION_NAMES[representation])
            & (master["classifier"] == master_label)
        ]
        master_f1 = round(float(match["macro_f1"].iloc[0]), 4) if len(match) else None

        result_id = session.next_result_id("classify")
        summary = {
            "result_id": result_id,
            "representation": representation,
            "classifier": classifier,
            "parameters": params,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "per_class_f1": {label: round(float(v), 4) for label, v in zip(labels, per_class)},
            "train_seconds": round(seconds, 2),
            "master_reference_macro_f1": master_f1,
        }
        full_report = {
            "y_true": y_test,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
            "report": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
        }
        session.store_result(
            "run_classifier",
            {"representation": representation, "classifier": classifier, **params},
            summary,
            full_report=full_report,
        )
        return summary

    @tool
    def show_misclassified_tool(
        result_id: str,
        n: int = 5,
        true_label: Optional[str] = None,
        predicted_label: Optional[str] = None,
    ) -> dict:
        """Show test tweets one classifier run got wrong, with their true and
        predicted emotion and the annotated intensity of the true emotion
        (0 to 1, low meaning weakly present). Sends real tweets, so at
        most 10, drawn by a fixed sample.

        Args:
            result_id: from run_classifier_tool, e.g. "classify_3".
            n: how many to show, 1 to 10.
            true_label: keep only this true emotion (anger, fear, joy,
                sadness). Optional.
            predicted_label: keep only this predicted emotion. Optional.
        """
        # For the reader, not the model: the sample is fixed by seed, so the
        # same call always shows the same tweets, and true_label with
        # predicted_label together isolate one confusion, e.g. fear read as
        # anger.

        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        entry = session.results_store.get(result_id)
        if entry is None or entry["record"].tool_name != "run_classifier":
            return {"error": f"'{result_id}' is not a result_id from run_classifier_tool."}
        labels = workspace["labels"]
        for value in (true_label, predicted_label):
            if value is not None and value not in labels:
                return {"error": f"Unknown emotion '{value}'. Use one of {labels}."}

        test = workspace["test"].copy()
        test["predicted"] = entry["full_report"]["y_pred"]
        errors = test[test["emotion"] != test["predicted"]]
        if true_label is not None:
            errors = errors[errors["emotion"] == true_label]
        if predicted_label is not None:
            errors = errors[errors["predicted"] == predicted_label]
        n = max(1, min(int(n), MAX_DOCUMENTS_PER_CALL))
        shown = errors.sample(n=min(n, len(errors)), random_state=SEED).sort_index() if len(errors) else errors

        new_id = session.next_result_id("misclassified")
        summary = {
            "result_id": new_id,
            "source_result_id": result_id,
            "filter": {"true_label": true_label, "predicted_label": predicted_label},
            "n_matching_errors": int(len(errors)),
            "n_shown": int(len(shown)),
            "documents": [
                {
                    "doc_id": int(row["id"]),
                    "true_emotion": str(row["emotion"]),
                    "predicted_emotion": str(row["predicted"]),
                    "intensity": round(float(row["intensity"]), 3),
                    "text": str(row["text"]),
                }
                for _, row in shown.iterrows()
            ],
        }
        session.store_result(
            "show_misclassified",
            {"result_id": result_id, "n": n, "true_label": true_label, "predicted_label": predicted_label},
            summary,
        )
        return summary

    return [run_classifier_tool, show_misclassified_tool]
