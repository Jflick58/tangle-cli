"""`tangle sdk pipelines dehydrate` drives the shared dehydrator end to end.

The API client is the REAL ``TangleApiClient`` stubbed only at its HTTP seams,
so publication verification and inspector-backed NAME exercise production
row/spec shapes rather than pre-enriched fixtures.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from tangle_cli import cli, pipelines_cli
from tangle_cli.utils import compute_spec_digest, dump_yaml

LEAF = {"name": "L", "implementation": {"container": {"image": "i", "command": ["c"]}}}
DIGEST = compute_spec_digest(LEAF)


def _write_pipeline(tmp_path: Path) -> Path:
    source = tmp_path / "in.yaml"
    source.write_text(
        dump_yaml({"name": "O", "implementation": {"graph": {"tasks": {"t": {"componentRef": {"spec": LEAF}}}}}}),
        encoding="utf-8",
    )
    return source


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Configure what the stubbed real client publishes; records CLI plumbing."""
    from tangle_cli.client import TangleApiClient
    from tangle_cli.models import ComponentSpec

    state: dict[str, Any] = {"specs": {}, "rows": [], "error": None, "kwargs": None}

    def factory(**kwargs: Any) -> Any:
        state["kwargs"] = kwargs
        client = TangleApiClient(base_url="https://review.invalid", include_env_credentials=False)

        def get_component_spec(digest: str) -> Any:
            if state["error"] is not None:
                raise state["error"]
            if digest not in state["specs"]:
                raise KeyError(digest)
            return ComponentSpec.from_dict(copy.deepcopy(state["specs"][digest]))

        client.get_component_spec = get_component_spec
        client._published_component_rows = lambda **kw: [
            dict(row) for row in state["rows"] if kw.get("digest") in (None, row["digest"])
        ]
        return client

    monkeypatch.setattr(pipelines_cli, "LazyTangleApiClient", factory)
    return state


def _run(args: list[str]) -> None:
    cli.build_app()(["sdk", "pipelines", "dehydrate", *args])


def _succeed(args: list[str]) -> None:
    try:
        _run(args)
    except SystemExit as exc:
        assert exc.code in (0, None), exc.code


def _task_ref(output: Path) -> dict[str, Any]:
    return yaml.safe_load(output.read_text(encoding="utf-8"))["implementation"]["graph"]["tasks"]["t"]["componentRef"]


def _fragment(output: Path) -> list[dict[str, Any]]:
    url = _task_ref(output)["url"]
    manifest = yaml.safe_load(output.with_name(f"{output.stem}.components.yaml").read_text(encoding="utf-8"))
    return manifest[url.split("#", 1)[1]]


def test_default_auto_emits_a_verified_published_digest(tmp_path: Path, library: dict[str, Any], capsys) -> None:
    library["specs"][DIGEST] = LEAF
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output), "--base-url", "https://x.test", "--token", "t"])

    assert _task_ref(output) == {"digest": DIGEST}
    assert "(mode: auto)" in capsys.readouterr().out
    assert library["kwargs"]["base_url"] == "https://x.test"
    assert library["kwargs"]["token"] == "t"


def test_default_auto_extracts_an_unpublished_component(tmp_path: Path, library: dict[str, Any], capsys) -> None:
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output)])

    assert _task_ref(output)["url"].startswith("file://./components/")
    assert "Components: 1 file(s)" in capsys.readouterr().out


def test_digest_mode_writes_a_portable_resolve_config(tmp_path: Path, library: dict[str, Any], capsys) -> None:
    library["specs"][DIGEST] = LEAF
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output), "--mode", "digest"])

    fragment = _fragment(output)
    assert fragment[0] == {"digest": DIGEST, "fallback_on_error": True}
    assert (tmp_path / fragment[1]["local"]).is_file()
    assert "Resolve config:" in capsys.readouterr().out


def test_name_mode_pins_the_inspected_owner(tmp_path: Path, library: dict[str, Any]) -> None:
    library["specs"][DIGEST] = LEAF
    library["rows"].append({"name": "L", "digest": DIGEST, "published_by": "user-123"})
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output), "--mode", "name"])

    assert _fragment(output)[0] == {"name": "L", "publisher": "user-123", "fallback_on_error": True}


def test_an_unreachable_library_degrades_to_the_local_copy(tmp_path: Path, library: dict[str, Any]) -> None:
    library["error"] = requests.ConnectionError("unreachable")
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output), "--mode", "digest"])

    fragment = _fragment(output)
    assert len(fragment) == 1 and set(fragment[0]) == {"local"}


def test_components_dir_override_is_honored(tmp_path: Path, library: dict[str, Any]) -> None:
    output = tmp_path / "out.yaml"

    _succeed([str(_write_pipeline(tmp_path)), "-o", str(output), "--mode", "file", "--components-dir", str(tmp_path / "bundle")])

    assert list((tmp_path / "bundle").glob("*.yaml"))
    assert not (tmp_path / "components").exists()


@pytest.mark.parametrize(
    ("label", "args"),
    [
        ("unknown-mode", ["--mode", "bogus"]),
        ("missing-output", None),
        ("missing-input", ["--missing-input"]),
    ],
)
def test_invalid_invocations_exit_non_zero_without_writing(
    tmp_path: Path, library: dict[str, Any], label: str, args: list[str] | None
) -> None:
    source = _write_pipeline(tmp_path)
    output = tmp_path / "out.yaml"
    if args is None:
        argv = [str(source)]
    elif args == ["--missing-input"]:
        argv = [str(tmp_path / "nope.yaml"), "-o", str(output)]
    else:
        argv = [str(source), "-o", str(output), *args]

    with pytest.raises(SystemExit) as excinfo:
        _run(argv)

    assert excinfo.value.code not in (0, None), label
    assert not output.exists(), label


def test_a_malformed_header_fails_loudly_instead_of_degrading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Misconfiguration is not an outage: it must not silently ship local-only."""
    monkeypatch.setenv("TANGLE_API_URL", "https://review.invalid")
    output = tmp_path / "out.yaml"

    with pytest.raises(SystemExit) as excinfo:
        _run([str(_write_pipeline(tmp_path)), "-o", str(output), "--mode", "digest", "--header", "no-colon"])

    assert excinfo.value.code not in (0, None)
    assert "header" in str(excinfo.value.code).lower()


# ------------------------------------------------------------ nested graphs


def _nested(leaf: dict[str, Any]) -> dict[str, Any]:
    inner = {"name": "Inner", "implementation": {"graph": {"tasks": {"leaf": {"componentRef": {"spec": leaf}}}}}}
    return {"name": "O", "implementation": {"graph": {"tasks": {"sub": {"componentRef": {"spec": inner}}}}}}


def _leaf_digests(document: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for task in document["implementation"]["graph"]["tasks"].values():
        spec = task["componentRef"]["spec"]
        if "graph" in spec["implementation"]:
            found += _leaf_digests(spec)
        else:
            found.append(compute_spec_digest({k: v for k, v in spec.items() if not k.startswith("_")}))
    return found


@pytest.mark.parametrize("mode", ["file", "url"])
def test_a_nested_graph_is_fully_dehydrated_portable_and_rehydrates(
    tmp_path: Path, library: dict[str, Any], mode: str
) -> None:
    """A public dehydrate must never return a partially hydrated document.

    The whole bundle is relocated before rehydrating, offline, so every ref
    written -- root, extracted subgraph and resolve config -- must be relative.
    """
    import shutil

    from tangle_cli.pipeline_hydrator import PipelineHydrator
    from tangle_cli.schema_validation import validate_dehydrated_pipeline

    work = tmp_path / "work"
    work.mkdir()
    document = _nested(LEAF)
    (work / "in.yaml").write_text(dump_yaml(document), encoding="utf-8")
    _succeed([str(work / "in.yaml"), "-o", str(work / "out.yaml"), "--mode", mode])

    written = [p for p in work.rglob("*.yaml") if p.name not in {"in.yaml", "out.components.yaml"}]
    for path in written:
        tasks = yaml.safe_load(path.read_text(encoding="utf-8")).get("implementation", {}).get("graph", {}).get("tasks", {})
        for task in tasks.values():
            assert "spec" not in task["componentRef"], f"inline spec left in {path.name}"
    validate_dehydrated_pipeline(yaml.safe_load((work / "out.yaml").read_text(encoding="utf-8")))

    moved = tmp_path / "moved"
    shutil.copytree(work, moved)
    shutil.rmtree(work)

    class Offline:
        def resolve_digest(self, digest: str) -> str:
            return digest

        def get_component_spec(self, digest: str) -> Any:
            raise requests.ConnectionError("offline")

    hydrated = PipelineHydrator(client=Offline()).hydrate_file(moved / "out.yaml").data
    assert _leaf_digests(hydrated) == _leaf_digests(document)


@pytest.mark.parametrize(("mode", "kept_url"), [("url", True), ("file", False)])
def test_leaf_mode_behavior_holds_inside_an_extracted_subgraph(
    tmp_path: Path, library: dict[str, Any], mode: str, kept_url: bool
) -> None:
    """Extracting the boundary must not change how the leaves inside it are
    replaced: url keeps a canonical URL, file always writes a local file."""
    canonical = "https://example.test/leaf.yaml"
    leaf = {**LEAF, "metadata": {"annotations": {"canonical_location": canonical}}}
    (tmp_path / "in.yaml").write_text(dump_yaml(_nested(leaf)), encoding="utf-8")

    _succeed([str(tmp_path / "in.yaml"), "-o", str(tmp_path / "out.yaml"), "--mode", mode])

    (subgraph,) = (tmp_path / "subgraphs").glob("*.yaml")
    leaf_ref = yaml.safe_load(subgraph.read_text(encoding="utf-8"))["implementation"]["graph"]["tasks"]["leaf"]["componentRef"]
    if kept_url:
        assert leaf_ref == {"url": canonical}
    else:
        assert leaf_ref["url"].startswith("file://") and leaf_ref["url"] != canonical


# ---------------------------------------------------------- input boundaries


@pytest.mark.parametrize("text", ["null", "false", "0", "''", "[]", "just a string"])
def test_a_non_mapping_pipeline_is_refused_without_writing(
    tmp_path: Path, library: dict[str, Any], monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Falsey YAML must not be read as an empty pipeline and written as ``{}``."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.yaml").write_text(text + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run(["in.yaml", "-o", "out.yaml"])

    assert excinfo.value.code not in (0, None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.yaml"]


@pytest.mark.parametrize("uri", ["gs://bucket/x.yaml", "https://example.test/x.yaml", "file:///tmp/x.yaml"])
@pytest.mark.parametrize("position", ["input", "output", "components-dir"])
def test_a_uri_is_refused_before_path_can_mangle_it_into_a_local_write(
    tmp_path: Path, library: dict[str, Any], monkeypatch: pytest.MonkeyPatch, uri: str, position: str
) -> None:
    """Path("gs://b/x") is the LOCAL path gs:/b/x; the command must refuse the
    URI instead of silently writing a gs:/... tree under the cwd."""
    monkeypatch.chdir(tmp_path)
    _write_pipeline(tmp_path)
    argv = {
        "input": [uri, "-o", "out.yaml"],
        "output": ["in.yaml", "-o", uri],
        "components-dir": ["in.yaml", "-o", "out.yaml", "--mode", "file", "--components-dir", uri],
    }[position]

    with pytest.raises(SystemExit) as excinfo:
        _run(argv)

    assert excinfo.value.code not in (0, None)
    assert "local path" in str(excinfo.value.code)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.yaml"]


@pytest.mark.parametrize("explicit", [True, False])
def test_the_standalone_helper_verifies_against_an_explicit_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """With no client injected, the lazily created client must use the
    requested endpoint, not the ambient TANGLE_API_URL."""
    from tangle_cli.pipelines import dehydrate_pipeline_file

    monkeypatch.setenv("TANGLE_API_URL", "https://ambient.invalid")
    hosts: list[str] = []

    def record(self: Any, method: str, url: str, **kwargs: Any) -> Any:
        hosts.append(url.split("/")[2])
        # A real, non-retryable 404: a ConnectionError would be retried with
        # backoff until the client's deadline (about a minute).
        response = requests.Response()
        response.status_code = 404
        response.url = url
        response.headers["Content-Type"] = "application/json"
        response._content = b'{"detail": "not found"}'
        return response

    monkeypatch.setattr(requests.Session, "request", record)

    dehydrate_pipeline_file(
        _write_pipeline(tmp_path),
        output=tmp_path / "out.yaml",
        mode="digest",
        base_url="https://requested.invalid" if explicit else None,
    )

    assert hosts, "verification never reached the client"
    assert set(hosts) == {"requested.invalid" if explicit else "ambient.invalid"}
