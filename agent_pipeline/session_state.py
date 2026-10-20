"""Mutable session state shared by all agent tools within one student session.

This is kept out of the LangGraph message state on purpose: intermediate
artifacts such as the raw dataset, the DataFrame, and the document-term
matrix can be large and aren't something the LLM should ever see
directly. Tools read and write this object by closure, and only ever
hand the LLM a small structured summary.

State spans the whole Lab 1 pipeline, from data loading through feature
filtering, organized into three parts: named pipeline slots
(`dataframe`, `feature_matrix`, `feature_names`, `labels`, and so on) for
the artifacts most tools build on, set as the pipeline progresses; an
`artifacts` dict as a catch-all for anything else a stage produces, such
as token lists, frequent patterns, an augmented matrix, reduced
coordinates, or cosine similarity results, referenced by name; and
`results_store`, the LLM-facing side of every tool call, holding small
structured summaries keyed by an explicit `result_id` the student can
reference in a later turn.
"""

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class ResultRecord:
    result_id: str
    tool_name: str
    args: dict
    summary: dict
    created_at: str


class SessionState:
    def __init__(self):
        # Pipeline slots -- populated progressively as the student works
        # through the lab; None until the relevant stage has run.
        self.raw_dataset = None       # sklearn Bunch from fetch_20newsgroups
        self.dataframe = None         # pandas DataFrame: text, category, category_name, [unigrams]
        self.feature_matrix = None    # document-term matrix (sparse), set by build_dtm
        self.feature_names = None     # vocabulary, aligned with feature_matrix columns
        self.labels = None            # np.array of category_name, aligned with feature_matrix rows
        self.categories: list = []    # sorted unique labels, kept in sync via set_labels()

        # Catch-all for everything else a stage produces.
        self.artifacts: dict[str, Any] = {}

        self.results_store: dict[str, dict] = {}
        self.pending_figure: Optional[Any] = None
        self._counter = itertools.count(1)

        # Which of the student's own workspace tools the agent is
        # actually allowed to run, and the pending batch request (if
        # any) awaiting the student's approval before all of them can
        # be granted at once. See tool_access.py: discovering a tool
        # and being told about it never grants it by itself.
        self.granted_tools: set[str] = set()
        self.pending_tool_request: Optional[dict] = None

    def set_labels(self, labels) -> None:
        self.labels = np.asarray(labels)
        self.categories = sorted(set(self.labels.tolist()))

    def next_result_id(self, tool_prefix: str) -> str:
        return f"{tool_prefix}_{next(self._counter)}"

    def store_result(self, tool_name: str, args: dict, summary: dict, full_report=None) -> str:
        result_id = summary["result_id"]
        self.results_store[result_id] = {
            "record": ResultRecord(
                result_id=result_id,
                tool_name=tool_name,
                args=args,
                summary=summary,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            "full_report": full_report,
        }
        return result_id
