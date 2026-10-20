"""Inspection tools: list_representations_tool, show_master_results_tool,
inspect_result_tool, and compare_results_tool.

These are usable immediately, without the request/grant gate: they only
read what is already saved
(lab2_workspace/ and earlier results in this session), compute nothing
new, and never send tweets to the language model, only schema and
numbers.
"""

from typing import List

from langchain_core.tools import tool

from lab2_tools.workspace import load_workspace, workspace_error


def make_tools(session):
    @tool
    def list_representations_tool() -> dict:
        """List what the Master saved: the five representations (name, sparse
        or dense, n_features, and what each supports), the dataset schema
        with row and label counts, and when it was saved. No tweets.
        """
        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        manifest = workspace["manifest"]
        train = workspace["train"]

        result_id = session.next_result_id("workspace")
        summary = {
            "result_id": result_id,
            "saved_at": manifest["created_at"],
            "representations": [
                {
                    "name": name,
                    "kind": spec["kind"],
                    "n_features": spec["n_features"],
                    "classification": True,
                    "retrieval": spec["retrieval"],
                }
                for name, spec in manifest["representations"].items()
            ],
            "dataset_schema": [{"column": c, "dtype": str(t)} for c, t in train.dtypes.items()],
            "n_train": int(len(train)),
            "n_test": int(len(workspace["test"])),
            "train_label_counts": {k: int(v) for k, v in train["emotion"].value_counts().items()},
        }
        session.store_result("list_representations", {}, summary)
        return summary

    @tool
    def show_master_results_tool(top_n: int = 17) -> dict:
        """The Master notebook's results table (representation, classifier,
        accuracy, macro F1), best macro F1 first. The baseline any new
        run_classifier_tool result should be compared against.

        Args:
            top_n: how many rows, 1 to 17 (the full table).
        """
        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        table = workspace["master_results"].sort_values("macro_f1", ascending=False)
        top_n = max(1, min(int(top_n), len(table)))

        result_id = session.next_result_id("master_results")
        summary = {
            "result_id": result_id,
            "rows": [
                {
                    "representation": row["representation"],
                    "classifier": row["classifier"],
                    "accuracy": round(float(row["accuracy"]), 4),
                    "macro_f1": round(float(row["macro_f1"]), 4),
                }
                for _, row in table.head(top_n).iterrows()
            ],
        }
        session.store_result("show_master_results", {"top_n": top_n}, summary)
        return summary

    @tool
    def inspect_result_tool(result_id: str) -> dict:
        """Break down one run_classifier_tool result: precision, recall, F1
        and support per emotion, the confusion matrix (rows true, columns
        predicted), and the most frequent confusion. Numbers only.

        Args:
            result_id: from run_classifier_tool, e.g. "classify_1".
        """
        try:
            workspace = load_workspace(session)
        except FileNotFoundError as exc:
            return workspace_error(exc)
        entry = session.results_store.get(result_id)
        if entry is None or entry["record"].tool_name != "run_classifier":
            return {"error": f"'{result_id}' is not a result_id from run_classifier_tool."}
        labels = workspace["labels"]
        report = entry["full_report"]["report"]
        cm = entry["full_report"]["confusion_matrix"]

        worst = (0, None, None)
        for i, true in enumerate(labels):
            for j, pred in enumerate(labels):
                if i != j and cm[i][j] > worst[0]:
                    worst = (int(cm[i][j]), true, pred)

        new_id = session.next_result_id("inspect")
        record = entry["record"].summary
        summary = {
            "result_id": new_id,
            "source_result_id": result_id,
            "representation": record["representation"],
            "classifier": record["classifier"],
            "macro_f1": record["macro_f1"],
            "per_class": [
                {
                    "emotion": label,
                    "precision": round(float(report[label]["precision"]), 4),
                    "recall": round(float(report[label]["recall"]), 4),
                    "f1": round(float(report[label]["f1-score"]), 4),
                    "support": int(report[label]["support"]),
                }
                for label in labels
            ],
            "confusion_matrix": [
                {"true_emotion": true, **{pred: int(cm[i][j]) for j, pred in enumerate(labels)}}
                for i, true in enumerate(labels)
            ],
            "most_confused": {"true_emotion": worst[1], "predicted_emotion": worst[2], "count": worst[0]},
        }
        session.store_result("inspect_result", {"result_id": result_id}, summary)
        return summary

    @tool
    def compare_results_tool(result_ids: List[str]) -> dict:
        """Put two or more run_classifier_tool results side by side with their
        parameters, accuracy and macro F1, best first, plus the gap
        between best and worst macro F1. Numbers only.

        Args:
            result_ids: two or more ids, e.g. ["classify_1", "classify_2"].
        """
        rows, problems = [], []
        for rid in result_ids:
            entry = session.results_store.get(rid)
            if entry is None or entry["record"].tool_name != "run_classifier":
                problems.append(rid)
                continue
            s = entry["record"].summary
            rows.append(
                {
                    "result_id": rid,
                    "representation": s["representation"],
                    "classifier": s["classifier"],
                    "parameters": str(s["parameters"]),
                    "accuracy": s["accuracy"],
                    "macro_f1": s["macro_f1"],
                }
            )
        if len(rows) < 2:
            return {"error": "Need at least two run_classifier_tool result_ids.", "not_found": problems}
        rows.sort(key=lambda r: r["macro_f1"], reverse=True)

        new_id = session.next_result_id("compare")
        summary = {
            "result_id": new_id,
            "rows": rows,
            "best_result_id": rows[0]["result_id"],
            "macro_f1_gap": round(rows[0]["macro_f1"] - rows[-1]["macro_f1"], 4),
            "not_found": problems,
        }
        session.store_result("compare_results", {"result_ids": result_ids}, summary)
        return summary

    return [list_representations_tool, show_master_results_tool, inspect_result_tool, compare_results_tool]
