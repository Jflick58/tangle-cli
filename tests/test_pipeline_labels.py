"""``@pipeline(labels=...)`` writes root ``metadata.labels``.

``GraphBuilder`` carried only annotations, so the five corpus pipelines with
a root ``metadata.labels`` block could not be authored in Python. Both
schemas type ``labels`` as ``additionalProperties: {"type": "string"}`` —
narrower than ``annotations``, which also admits numbers, booleans and null
— so the value policy here is strict ``str -> str``.

Labels are descriptive: nothing in the CLI, hydrator or client reads them.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from tangle_cli.pipeline_compiler import compile_pipeline
from tangle_cli.python_pipeline import pipeline
from tangle_cli.python_pipeline.errors import InvalidPipelineLabelsError

# The exact label blocks carried by the five corpus pipelines.
_CORPUS_LABELS = {
    "search_signals": {
        "team": "discovery",
        "domain": "search-signals",
        "stage": "analysis",
    },
    "join_features": {
        "team": "discovery",
        "domain": "storefront-reranker",
        "stage": "experiment",
    },
    "l1_tangentable": {
        "team": "discovery",
        "stage": "experiment",
        "domain": "shop-app-search-ranking",
        "tangentable": "v3",
    },
    "smoke_test": {
        "team": "discovery",
        "domain": "storefront-reranker",
        "stage": "smoke-test",
    },
}

_TEMPLATE = '''
from tangle_cli.python_pipeline import Out, pipeline, task


@task(image="python:3.12")
def echo(value: str) -> str:
    return value


@pipeline("Labelled"__DECORATOR_ARGS__)
def labelled() -> Out[str]:
    run_it = echo.named("Echo")(value="x")
    return run_it
'''


def _compile(tmp_path: Path, decorator_args: str, case: str) -> dict:
    case_dir = tmp_path / case
    case_dir.mkdir(parents=True, exist_ok=True)
    script = case_dir / "pipeline.py"
    script.write_text(
        textwrap.dedent(_TEMPLATE).replace("__DECORATOR_ARGS__", decorator_args),
        encoding="utf-8",
    )
    out = case_dir / "compiled.yaml"
    compile_pipeline(script, out)
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def _normalize(path: Path) -> str:
    """Compiled text with the child sidecar's content hash masked, so two
    compiles in different directories can be compared."""
    return re.sub(r"child-[0-9a-f]{8}", "child-HASH", path.read_text(encoding="utf-8"))


def _child_sidecars(case_dir: Path) -> list[Path]:
    """Child subgraph YAMLs, excluding the sibling ``.components.yaml``."""
    return sorted(
        path
        for path in (case_dir / "compiled.subgraphs").glob("child-*.yaml")
        if not path.name.endswith(".components.yaml")
    )


# ============================================================================
# Emitted shape
# ============================================================================


@pytest.mark.parametrize("case, labels", sorted(_CORPUS_LABELS.items()))
def test_each_corpus_label_block_round_trips(tmp_path, case, labels):
    doc = _compile(tmp_path, f", labels={labels!r}", case)

    assert doc["metadata"] == {"labels": labels}
    # Key order inside the block is the author's, not sorted.
    assert list(doc["metadata"]["labels"]) == list(labels)


def test_labels_are_written_before_annotations(tmp_path):
    """Corpus metadata blocks are 4:1 labels-first, so that is the canonical
    order regardless of which keyword the author passed first."""
    doc = _compile(
        tmp_path,
        ', annotations={"version": "1.0"}, labels={"team": "discovery"}',
        "both",
    )

    assert list(doc["metadata"]) == ["labels", "annotations"]
    assert doc["metadata"] == {
        "labels": {"team": "discovery"},
        "annotations": {"version": "1.0"},
    }


def test_labels_reach_the_yaml_text_as_plain_strings(tmp_path):
    """Guards the rendered text, not just the parsed mapping."""
    case_dir = tmp_path / "text"
    case_dir.mkdir()
    script = case_dir / "pipeline.py"
    script.write_text(
        textwrap.dedent(_TEMPLATE).replace(
            "__DECORATOR_ARGS__", ', labels={"team": "discovery", "stage": "analysis"}'
        ),
        encoding="utf-8",
    )
    out = case_dir / "compiled.yaml"
    compile_pipeline(script, out)

    text = out.read_text(encoding="utf-8")
    assert "metadata:\n  labels:\n    team: discovery\n    stage: analysis\n" in text


# ============================================================================
# No labels changes nothing
# ============================================================================


def test_a_pipeline_without_labels_is_byte_identical(tmp_path):
    """The new block must not perturb any pipeline that does not use it."""
    without = _compile(tmp_path, "", "plain_a")
    explicit_empty = _compile(tmp_path, ", labels={}", "plain_b")
    explicit_none = _compile(tmp_path, ", labels=None", "plain_c")

    assert "metadata" not in without
    assert without == explicit_empty == explicit_none

    a = (tmp_path / "plain_a" / "compiled.yaml").read_bytes()
    b = (tmp_path / "plain_b" / "compiled.yaml").read_bytes()
    c = (tmp_path / "plain_c" / "compiled.yaml").read_bytes()
    assert a == b == c


def test_annotations_alone_still_emit_the_same_metadata_block(tmp_path):
    doc = _compile(tmp_path, ', annotations={"version": "1.0"}', "ann_only")

    assert doc["metadata"] == {"annotations": {"version": "1.0"}}


# ============================================================================
# Subpipeline
# ============================================================================


def test_labels_do_not_inherit_into_a_subpipeline_child(tmp_path):
    """Consistent with root annotations: a child keeps exactly what its own
    ``@pipeline`` declared, so child sidecar bytes are unaffected."""
    case_dir = tmp_path / "subpipe"
    case_dir.mkdir()
    script = case_dir / "pipeline.py"
    script.write_text(
        textwrap.dedent(
            '''
            from tangle_cli.python_pipeline import In, Out, pipeline, subpipeline, task


            @task(image="python:3.12")
            def echo(value: str) -> str:
                return value


            @pipeline("Child")
            def child(seed: In[str]) -> Out[str]:
                inner = echo.named("Inner")(value=seed)
                return inner


            @pipeline("Parent", labels={"team": "discovery"})
            def parent() -> Out[str]:
                kid = subpipeline(child).named("Run child")(seed="x")
                return kid.wait_for_output
            '''
        ),
        encoding="utf-8",
    )
    out = case_dir / "compiled.yaml"
    compile_pipeline(script, out, pipeline_name="parent")

    parent_doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert parent_doc["metadata"] == {"labels": {"team": "discovery"}}

    children = _child_sidecars(case_dir)
    assert len(children) == 1
    child_doc = yaml.safe_load(children[0].read_text(encoding="utf-8"))
    assert "metadata" not in child_doc

    # Stronger than "no metadata key": compile the same pair with no labels
    # at all and compare. The child sidecar's name is a content hash that
    # also folds in the output directory, so the two compiles necessarily
    # live in different directories and the hash token is normalized away.
    # Everything else must match, in BOTH documents.
    unlabelled_parent, unlabelled_child = _compile_parent_without_labels(tmp_path)
    assert _normalize(children[0]) == _normalize(unlabelled_child)
    assert _normalize(out) == _normalize(unlabelled_parent).replace(
        "name: Parent\n", "name: Parent\nmetadata:\n  labels:\n    team: discovery\n"
    )


def test_a_labelled_child_keeps_its_own_labels(tmp_path):
    """The other direction: declaring labels on the child works, and does not
    leak up to the parent."""
    case_dir = tmp_path / "subpipe_child"
    case_dir.mkdir()
    script = case_dir / "pipeline.py"
    script.write_text(
        textwrap.dedent(
            '''
            from tangle_cli.python_pipeline import In, Out, pipeline, subpipeline, task


            @task(image="python:3.12")
            def echo(value: str) -> str:
                return value


            @pipeline("Child", labels={"stage": "analysis"})
            def child(seed: In[str]) -> Out[str]:
                inner = echo.named("Inner")(value=seed)
                return inner


            @pipeline("Parent")
            def parent() -> Out[str]:
                kid = subpipeline(child).named("Run child")(seed="x")
                return kid.wait_for_output
            '''
        ),
        encoding="utf-8",
    )
    out = case_dir / "compiled.yaml"
    compile_pipeline(script, out, pipeline_name="parent")

    parent_doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "metadata" not in parent_doc

    child_doc = yaml.safe_load(
        _child_sidecars(case_dir)[0].read_text(encoding="utf-8")
    )
    assert child_doc["metadata"] == {"labels": {"stage": "analysis"}}


# ============================================================================
# Validation
# ============================================================================


def _reject(labels) -> str:
    with pytest.raises(InvalidPipelineLabelsError) as excinfo:

        @pipeline("Rejected", labels=labels)
        def _p():  # pragma: no cover - never traced
            pass

    return str(excinfo.value)


def test_a_non_string_value_is_refused():
    """Stricter than annotations, which accept numbers, booleans and null:
    both schemas type label values as string."""
    message = _reject({"replicas": 3})

    assert "labels value for key 'replicas' must be a string" in message
    assert "int" in message


def test_a_bool_value_is_refused():
    assert "must be a string" in _reject({"tangentable": True})


def test_a_none_value_is_refused():
    assert "must be a string" in _reject({"stage": None})


def test_a_non_string_key_is_refused():
    assert "keys must be strings" in _reject({3: "discovery"})


def test_an_empty_key_is_refused():
    assert "must not be empty" in _reject({"": "discovery"})


def test_a_non_mapping_is_refused():
    assert "must be a mapping" in _reject(["team=discovery"])


def test_a_template_delimiter_is_refused():
    """Moves the failure off the whole-output delimiter scan and onto a
    message that names the offending label key."""
    assert "team" in _reject({"team": "{{ team_name }}"})


def test_the_system_prefix_is_allowed_on_labels():
    """``system/`` is reserved for Tangle's own ANNOTATIONS. Nothing reserves
    a label prefix, so refusing it would reject a document both schemas
    accept."""

    @pipeline("Reserved-ish", labels={"system/owner": "discovery"})
    def _p():  # pragma: no cover - never traced
        pass

    assert _p.labels == {"system/owner": "discovery"}


def test_diagnostics_never_echo_a_label_value():
    for labels in (
        {"team": 99999},
        {"team": {"nested": "super-secret-label"}},
        {"team": "{{ super-secret-label }}"},
    ):
        message = _reject(labels)
        assert "super-secret-label" not in message
        assert "99999" not in message


def test_the_checked_mapping_is_copied():
    """A later mutation of the caller's dict must not reach the compile."""
    supplied = {"team": "discovery"}

    @pipeline("Copied", labels=supplied)
    def _p():  # pragma: no cover - never traced
        pass

    supplied["team"] = "changed"
    assert _p.labels == {"team": "discovery"}


def _compile_parent_without_labels(tmp_path: Path) -> tuple[Path, Path]:
    """Compile the same parent/child pair with no labels anywhere and return
    ``(parent, child)`` paths, for the leak comparison above."""
    case_dir = tmp_path / "subpipe_baseline"
    case_dir.mkdir()
    script = case_dir / "pipeline.py"
    script.write_text(
        textwrap.dedent(
            '''
            from tangle_cli.python_pipeline import In, Out, pipeline, subpipeline, task


            @task(image="python:3.12")
            def echo(value: str) -> str:
                return value


            @pipeline("Child")
            def child(seed: In[str]) -> Out[str]:
                inner = echo.named("Inner")(value=seed)
                return inner


            @pipeline("Parent")
            def parent() -> Out[str]:
                kid = subpipeline(child).named("Run child")(seed="x")
                return kid.wait_for_output
            '''
        ),
        encoding="utf-8",
    )
    out = case_dir / "compiled.yaml"
    compile_pipeline(script, out, pipeline_name="parent")
    return out, _child_sidecars(case_dir)[0]
