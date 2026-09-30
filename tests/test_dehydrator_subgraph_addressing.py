"""Extracted subgraph files are addressed by their written content.

A per-run counter (``<name>_0.yaml``) let a second pipeline dehydrated into the
same directory overwrite the first one's subgraph with different content, so
the first output silently rehydrated to the second's pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from tangle_cli.pipeline_dehydrator import DehydrateChoice, PipelineDehydrator
from tangle_cli.pipeline_hydrator import PipelineHydrator
from tangle_cli.pipelines import dehydrate_pipeline_file
from tangle_cli.utils import dump_yaml

MODES = ["auto", "file", "url"]


class NoLibrary:
    """Nothing is published, so every leaf ends up local."""

    def resolve_digest(self, digest: str) -> str:
        return digest

    def get_component_spec(self, digest: str) -> Any:
        raise KeyError(digest)

    def list_published_component_infos(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _leaf(image: str) -> dict[str, Any]:
    return {"name": "L", "implementation": {"container": {"image": image}}}


def _graph(name: str, tasks: dict[str, Any], annotations: dict[str, str] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": name, "implementation": {"graph": {"tasks": tasks}}}
    if annotations:
        spec["metadata"] = {"annotations": annotations}
    return spec


def _ref(spec: dict[str, Any]) -> dict[str, Any]:
    return {"componentRef": {"spec": spec}}


def _dehydrate(directory: Path, document: dict[str, Any], mode: str, out: str = "out.yaml") -> Path:
    source = directory / f"{out}.in"
    source.write_text(dump_yaml(document), encoding="utf-8")
    dehydrate_pipeline_file(source, output=directory / out, mode=mode, client=NoLibrary())
    return directory / out


def _rehydrate(output: Path) -> dict[str, Any]:
    return PipelineHydrator(client=NoLibrary()).hydrate_file(output).data


def _images(document: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for task in document["implementation"]["graph"]["tasks"].values():
        spec = task["componentRef"]["spec"]
        if "graph" in spec["implementation"]:
            found += _images(spec)
        else:
            found.append(spec["implementation"]["container"]["image"])
    return found


def _subgraph_files(directory: Path) -> list[str]:
    return sorted(p.name for p in (directory / "subgraphs").glob("*"))


@pytest.mark.parametrize("mode", MODES)
def test_identical_subgraphs_share_one_file_and_reference(tmp_path: Path, mode: str) -> None:
    judge = _graph("Judge", {"leaf": _ref(_leaf("i"))})
    out = _dehydrate(tmp_path, _graph("O", {"a": _ref(judge), "b": _ref(judge)}), mode)

    tasks = yaml.safe_load(out.read_text(encoding="utf-8"))["implementation"]["graph"]["tasks"]
    assert len(_subgraph_files(tmp_path)) == 1
    assert tasks["a"]["componentRef"] == tasks["b"]["componentRef"]


@pytest.mark.parametrize("mode", MODES)
def test_a_provenance_only_difference_keeps_two_files(tmp_path: Path, mode: str) -> None:
    """Like two generated sources of one Judge spec: different documents."""
    tasks = {"leaf": _ref(_leaf("i"))}
    first = _graph("Judge", tasks, {"component_yaml_path": "judge-47d488e5.yaml"})
    second = _graph("Judge", tasks, {"component_yaml_path": "judge-f89931a0.yaml"})

    _dehydrate(tmp_path, _graph("O", {"a": _ref(first), "b": _ref(second)}), mode)

    assert len(_subgraph_files(tmp_path)) == 2


@pytest.mark.parametrize("mode", MODES)
def test_a_second_output_in_the_same_directory_cannot_mutate_the_first(tmp_path: Path, mode: str) -> None:
    def pipeline(image: str) -> dict[str, Any]:
        return _graph("O", {"s": _ref(_graph("Judge", {"leaf": _ref(_leaf(image))}))})

    a = _dehydrate(tmp_path, pipeline("image-A"), mode, "a.yaml")
    b = _dehydrate(tmp_path, pipeline("image-B"), mode, "b.yaml")

    assert _images(_rehydrate(a)) == ["image-A"]
    assert _images(_rehydrate(b)) == ["image-B"]


@pytest.mark.parametrize("mode", MODES)
def test_deep_subgraphs_are_merkle_addressed_stable_and_relocatable(tmp_path: Path, mode: str) -> None:
    """A deep leaf change renames every enclosing subgraph; identical input
    reproduces identical names; the bundle rehydrates after relocation."""

    def pipeline(image: str) -> dict[str, Any]:
        inner = _graph("Inner", {"leaf": _ref(_leaf(image))})
        return _graph("O", {"mid": _ref(_graph("Mid", {"inner": _ref(inner)}))})

    names = {}
    for label, image in (("first", "x"), ("again", "x"), ("changed", "y")):
        directory = tmp_path / label
        directory.mkdir()
        _dehydrate(directory, pipeline(image), mode)
        names[label] = _subgraph_files(directory)

    assert names["first"] == names["again"]
    assert len(names["first"]) == 2
    assert not set(names["first"]) & set(names["changed"])

    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "first", moved)
    shutil.rmtree(tmp_path / "first")
    assert _images(_rehydrate(moved / "out.yaml")) == ["x"]


def test_a_reused_instance_keeps_no_stale_subgraph_map_and_honors_the_extension(tmp_path: Path) -> None:
    """Each output writes its own subgraph files, with the configured extension."""
    dehydrator = PipelineDehydrator({"": DehydrateChoice.FILE}, component_extension=".yml")
    judge = _graph("Judge", {"leaf": _ref(_leaf("i"))})
    for index in range(2):
        directory = tmp_path / f"out{index}"
        directory.mkdir()
        (directory / "in.yaml").write_text(dump_yaml(_graph("O", {"s": _ref(judge)})), encoding="utf-8")
        dehydrator.dehydrate_file(directory / "in.yaml", directory / "out.yaml")

        (only,) = _subgraph_files(directory)
        assert only.endswith(".yml")
        assert "../" not in (directory / "out.yaml").read_text(encoding="utf-8")


def test_a_tampered_file_under_a_content_addressed_name_is_rewritten(tmp_path: Path) -> None:
    """A name on disk is not trusted to still hold the content it addresses."""
    document = _graph("O", {"s": _ref(_graph("Judge", {"leaf": _ref(_leaf("i"))}))})
    out = _dehydrate(tmp_path, document, "file")
    (name,) = _subgraph_files(tmp_path)
    (tmp_path / "subgraphs" / name).write_text("name: tampered\n", encoding="utf-8")

    _dehydrate(tmp_path, document, "file")

    assert _images(_rehydrate(out)) == ["i"]
