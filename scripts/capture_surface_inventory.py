#!/usr/bin/env python3
"""Capture deterministic, offline inventories from genuine provider SDK clients.

Provider SDK imports live in this test-only script. Core Solwyn modules remain
duck-typed and provider-independent. The capture records public names and
structural types only; it never invokes provider operations or reads request,
prompt, response, or credential content.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import re
import socket
import warnings
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

from solwyn._surface_graph import (
    SurfaceInspectionError,
    declared_namespace_paths,
    observe_public_surface,
)
from solwyn._surfaces import DIALECT_BY_PROVIDER, SurfaceContext

SCHEMA_VERSION = 1
FINGERPRINT_SCHEMA_VERSION = 1
ROOT = Path(__file__).parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "build" / "provider_surface_inventory"
DEFAULT_FINGERPRINT_PATH = ROOT / "tests" / "provider_surface_fingerprints.json"
INTERVAL_CATALOG_PATH = ROOT / "tests" / "provider_surface_intervals.json"
_INTERVAL_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


class ShapeSpec(NamedTuple):
    key: str
    family: str
    distribution: str
    client_shape: str
    provider: str
    mode: str
    variant: str


class InventoryRun(NamedTuple):
    paths: tuple[Path, ...]
    mismatches: tuple[str, ...]


_SHAPES = (
    ShapeSpec(
        "anthropic_async", "anthropic", "anthropic", "anthropic_sdk", "anthropic", "async", "native"
    ),
    ShapeSpec(
        "anthropic_sync", "anthropic", "anthropic", "anthropic_sdk", "anthropic", "sync", "native"
    ),
    ShapeSpec(
        "azure_openai_async", "openai", "openai", "openai_sdk", "azure_openai", "async", "azure"
    ),
    ShapeSpec(
        "azure_openai_sync", "openai", "openai", "openai_sdk", "azure_openai", "sync", "azure"
    ),
    ShapeSpec(
        "bedrock_aioboto3_async",
        "bedrock",
        "aioboto3",
        "bedrock_aioboto3",
        "bedrock",
        "async",
        "native",
    ),
    ShapeSpec(
        "bedrock_boto3_sync",
        "bedrock",
        "boto3",
        "bedrock_boto3",
        "bedrock",
        "sync",
        "native",
    ),
    ShapeSpec(
        "google_genai_async",
        "google-genai",
        "google-genai",
        "google_genai",
        "google",
        "async",
        "aio",
    ),
    ShapeSpec(
        "google_genai_sync",
        "google-genai",
        "google-genai",
        "google_genai",
        "google",
        "sync",
        "native",
    ),
    ShapeSpec(
        "google_generativeai_sync",
        "google-generativeai",
        "google-generativeai",
        "google_generativeai",
        "google",
        "sync",
        "legacy",
    ),
    ShapeSpec(
        "openai_compatible_async",
        "openai",
        "openai",
        "openai_sdk",
        "openai_compatible",
        "async",
        "generic_compatible",
    ),
    ShapeSpec(
        "openai_compatible_sync",
        "openai",
        "openai",
        "openai_sdk",
        "openai_compatible",
        "sync",
        "generic_compatible",
    ),
    ShapeSpec("openai_native_async", "openai", "openai", "openai_sdk", "openai", "async", "native"),
    ShapeSpec("openai_native_sync", "openai", "openai", "openai_sdk", "openai", "sync", "native"),
    ShapeSpec(
        "openai_together_async",
        "openai",
        "openai",
        "openai_sdk",
        "together",
        "async",
        "openai_compatible",
    ),
    ShapeSpec(
        "openai_together_sync",
        "openai",
        "openai",
        "openai_sdk",
        "together",
        "sync",
        "openai_compatible",
    ),
    ShapeSpec(
        "together_native_async",
        "together",
        "together",
        "native_together",
        "together",
        "async",
        "native",
    ),
    ShapeSpec(
        "together_native_sync",
        "together",
        "together",
        "native_together",
        "together",
        "sync",
        "native",
    ),
)
_SHAPES_BY_KEY = {spec.key: spec for spec in _SHAPES}
_IMPORT_NAME_BY_DISTRIBUTION = {
    "google-genai": "google.genai",
    "google-generativeai": "google.generativeai",
}
_FAMILY_SHAPES = {
    family: frozenset(spec.key for spec in _SHAPES if spec.family == family)
    for family in sorted({spec.family for spec in _SHAPES})
}


class OfflineViolationError(RuntimeError):
    """Raised if client construction or observation attempts socket I/O."""


class _SocketCounter:
    def __init__(self) -> None:
        self.attempts = 0

    def deny(self, *args: object, **kwargs: object) -> None:
        self.attempts += 1
        raise OfflineViolationError("provider inventory attempted socket access")


def shape_keys() -> tuple[str, ...]:
    """Return the stable mandatory real-client shape keys."""

    return tuple(spec.key for spec in _SHAPES)


def shape_keys_for_families(families: Collection[str]) -> tuple[str, ...]:
    """Expand stable matrix-family names to exact real-client variants."""

    family_set = set(families)
    unknown = sorted(family_set - set(_FAMILY_SHAPES))
    if unknown:
        raise ValueError(f"unknown provider family: {unknown[0]}")
    return tuple(sorted({key for family in family_set for key in _FAMILY_SHAPES[family]}))


def import_name_for_shape(shape_key: str) -> str:
    """Return the provider module that must be installed for a shape."""

    try:
        distribution = _SHAPES_BY_KEY[shape_key].distribution
    except KeyError:
        raise ValueError(f"unknown shape key: {shape_key}") from None
    return _IMPORT_NAME_BY_DISTRIBUTION.get(distribution, distribution)


@contextmanager
def _deny_socket_access() -> Iterator[_SocketCounter]:
    counter = _SocketCounter()
    with (
        patch.object(socket, "create_connection", side_effect=counter.deny),
        patch.object(socket.socket, "connect", side_effect=counter.deny),
        patch.object(socket.socket, "connect_ex", side_effect=counter.deny),
    ):
        yield counter


def _make_regular_client(spec: ShapeSpec) -> object:
    if spec.key.startswith(("openai_", "azure_openai_")):
        import openai

        client_type = openai.AsyncOpenAI if spec.mode == "async" else openai.OpenAI
        kwargs: dict[str, object] = {"api_key": "sk-test"}
        if spec.variant == "generic_compatible":
            kwargs["base_url"] = "https://provider.invalid/v1"
        elif spec.variant == "openai_compatible":
            kwargs["base_url"] = "https://api.together.xyz/v1"
        if spec.variant == "azure":
            azure_type = openai.AsyncAzureOpenAI if spec.mode == "async" else openai.AzureOpenAI
            return azure_type(
                api_key="sk-test",
                api_version="2024-06-01",
                azure_endpoint="https://example.openai.azure.com",
            )
        return client_type(**kwargs)

    if spec.key.startswith("anthropic_"):
        import anthropic

        client_type = anthropic.AsyncAnthropic if spec.mode == "async" else anthropic.Anthropic
        return client_type(api_key="sk-ant-test")

    if spec.key.startswith("together_native_"):
        import together

        client_type = together.AsyncTogether if spec.mode == "async" else together.Together
        return client_type(api_key="test")

    if spec.key.startswith("google_genai_"):
        from google import genai

        client = genai.Client(api_key="test")
        return client.aio if spec.mode == "async" else client

    if spec.key == "google_generativeai_sync":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as legacy_genai

        return legacy_genai.GenerativeModel("gemini-test")

    if spec.key == "bedrock_boto3_sync":
        import boto3

        return boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )

    raise RuntimeError(f"no regular client constructor for {spec.key}")


@asynccontextmanager
async def _client_for(spec: ShapeSpec) -> AsyncIterator[object]:
    if spec.key == "bedrock_aioboto3_async":
        import aioboto3

        async with aioboto3.Session().client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        ) as client:
            yield client
        return

    client = _make_regular_client(spec)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def _distribution_rows(spec: ShapeSpec) -> list[dict[str, str]]:
    names = [spec.distribution]
    if spec.key == "bedrock_boto3_sync":
        names.append("botocore")
    elif spec.key == "bedrock_aioboto3_async":
        names.extend(("aiobotocore", "boto3", "botocore"))
    return [{"name": name, "version": version(name)} for name in names]


def _bedrock_service_model_operations(client: object) -> tuple[str, ...]:
    meta = getattr(client, "meta", None)
    service_model = getattr(meta, "service_model", None)
    operation_names = getattr(service_model, "operation_names", ())
    if not isinstance(operation_names, Collection) or isinstance(operation_names, (str, bytes)):
        raise SurfaceInspectionError("<root>", "service_model_operations")
    import botocore

    return tuple(sorted({botocore.xform_name(name) for name in operation_names}))


async def _capture_shape(spec: ShapeSpec, structural_interval: str) -> dict[str, Any]:
    with _deny_socket_access() as socket_counter:
        async with _client_for(spec) as client:
            context = SurfaceContext(
                provider=spec.provider,
                dialect=DIALECT_BY_PROVIDER[spec.provider],
                client_shape=spec.client_shape,
                mode=spec.mode,
            )
            declared = declared_namespace_paths(context)
            observed = observe_public_surface(
                client,
                namespaces=declared,
                require_all_namespaces=False,
            )
            observed_paths = {item.path for item in observed}
            namespaces = tuple(sorted(declared & observed_paths))
            rows: dict[str, dict[str, str]] = {
                item.path: {
                    "path": item.path,
                    "descriptor_category": item.descriptor_category,
                    "return_shape": item.return_shape,
                    "source": "public_attribute",
                }
                for item in observed
            }
            service_operations: tuple[str, ...] = ()
            if spec.client_shape.startswith("bedrock_"):
                service_operations = _bedrock_service_model_operations(client)
                for path in service_operations:
                    rows.setdefault(
                        path,
                        {
                            "path": path,
                            "descriptor_category": "service_model_operation",
                            "return_shape": "service_model_only",
                            "source": "service_model_operation",
                        },
                    )

        distributions = _distribution_rows(spec)
        return {
            "schema_version": SCHEMA_VERSION,
            "shape_key": spec.key,
            "client_shape": spec.client_shape,
            "provider": spec.provider,
            "mode": spec.mode,
            "variant": spec.variant,
            "structural_interval": structural_interval,
            "distribution": distributions[0],
            "distributions": distributions,
            "socket_attempts": socket_counter.attempts,
            "namespaces": list(namespaces),
            "service_model_operations": list(service_operations),
            "observations": [rows[path] for path in sorted(rows)],
        }


def _validate_interval(structural_interval: str) -> None:
    if not _INTERVAL_RE.fullmatch(structural_interval):
        raise ValueError("structural_interval must be a lowercase path-safe label")


def capture_all(
    *,
    structural_interval: str,
    selected: Collection[str] | None = None,
    on_report: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Capture selected mandatory shapes with all socket access denied."""

    _validate_interval(structural_interval)
    keys = set(shape_keys()) if selected is None else set(selected)
    unknown = sorted(keys - set(shape_keys()))
    if unknown:
        raise ValueError(f"unknown shape key: {unknown[0]}")

    async def capture() -> tuple[dict[str, Any], ...]:
        reports: list[dict[str, Any]] = []
        for key in sorted(keys):
            report = await _capture_shape(_SHAPES_BY_KEY[key], structural_interval)
            if on_report is not None:
                on_report(report)
            reports.append(report)
        return tuple(reports)

    return asyncio.run(capture())


def report_filename(report: Mapping[str, Any]) -> str:
    return f"{report['shape_key']}--{report['structural_interval']}.json"


def render_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _structural_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "shape_key": report["shape_key"],
        "client_shape": report["client_shape"],
        "provider": report["provider"],
        "mode": report["mode"],
        "variant": report["variant"],
        "structural_interval": report["structural_interval"],
        "namespaces": report["namespaces"],
        "service_model_operations": report["service_model_operations"],
        "observations": report["observations"],
    }


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("report schema_version is unsupported")
    if report.get("socket_attempts") != 0:
        raise RuntimeError("report socket_attempts must be zero")
    try:
        _structural_payload(report)
        report["distributions"]
    except KeyError as exc:
        raise RuntimeError(f"report missing required field: {exc.args[0]}") from None


def fingerprint_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize one inventory with a version-independent structural digest."""

    _validate_report(report)
    canonical = json.dumps(
        _structural_payload(report), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "shape_key": report["shape_key"],
        "client_shape": report["client_shape"],
        "provider": report["provider"],
        "mode": report["mode"],
        "variant": report["variant"],
        "structural_interval": report["structural_interval"],
        "distributions": report["distributions"],
        "namespace_count": len(report["namespaces"]),
        "observation_count": len(report["observations"]),
        "service_model_operation_count": len(report["service_model_operations"]),
        "structure_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _report_key(report: Mapping[str, Any]) -> tuple[str, str]:
    return str(report["shape_key"]), str(report["structural_interval"])


def _display_key(key: tuple[str, str]) -> str:
    return f"{key[0]}@{key[1]}"


def _load_fingerprint_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError(f"invalid fingerprint manifest: {path}")
    rows = manifest.get("fingerprints")
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid fingerprint manifest: {path}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid fingerprint manifest: {path}")
        try:
            key = _report_key(row)
        except KeyError:
            raise RuntimeError(f"invalid fingerprint manifest: {path}") from None
        if key in indexed:
            raise RuntimeError(f"duplicate fingerprint: {_display_key(key)}")
        indexed[key] = row
    return indexed


def _render_fingerprint_manifest(
    indexed: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    manifest = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "fingerprints": [indexed[key] for key in sorted(indexed)],
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def update_fingerprint_manifest(
    path: Path,
    reports: Collection[Mapping[str, Any]],
) -> None:
    """Replace fingerprints for supplied shape/interval pairs and preserve the rest."""

    indexed = _load_fingerprint_manifest(path)
    for report in reports:
        indexed[_report_key(report)] = fingerprint_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_fingerprint_manifest(indexed), encoding="utf-8")


def _comparison_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "distributions"}


def _fingerprint_drift_message(
    key: tuple[str, str],
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> str:
    counts = (
        "namespace_count",
        "observation_count",
        "service_model_operation_count",
    )
    rendered = "; ".join(
        f"{name} {expected[name]} -> {actual[name]}, "
        f"delta {int(actual[name]) - int(expected[name]):+d}"
        for name in counts
    )
    return f"fingerprint drift: {_display_key(key)} ({rendered})"


def compare_fingerprints(
    path: Path,
    reports: Collection[Mapping[str, Any]],
    *,
    require_all: bool = False,
) -> tuple[str, ...]:
    """Compare structural fingerprints while treating versions as audit metadata."""

    indexed = _load_fingerprint_manifest(path)
    mismatches: list[str] = []
    report_keys = {_report_key(report) for report in reports}
    if require_all:
        mismatches.extend(
            f"missing report: {_display_key(key)}" for key in sorted(set(indexed) - report_keys)
        )
    for report in sorted(reports, key=_report_key):
        key = _report_key(report)
        expected = indexed.get(key)
        if expected is None:
            mismatches.append(f"missing fingerprint: {_display_key(key)}")
            continue
        actual = fingerprint_report(report)
        if _comparison_row(expected) != _comparison_row(actual):
            mismatches.append(_fingerprint_drift_message(key, expected, actual))
    return tuple(mismatches)


def load_reports(directory: Path) -> tuple[dict[str, Any], ...]:
    """Load full inventory reports from an artifact directory."""

    reports: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise RuntimeError(f"invalid report: {path}")
        try:
            _validate_report(report)
            key = _report_key(report)
        except (KeyError, RuntimeError):
            raise RuntimeError(f"invalid report: {path}") from None
        if key in seen:
            raise RuntimeError(f"duplicate report: {_display_key(key)}")
        seen.add(key)
        reports.append(report)
    return tuple(sorted(reports, key=_report_key))


def select_reports(
    reports: Collection[Mapping[str, Any]],
    *,
    selected: Collection[str] | None,
    structural_interval: str | None,
) -> tuple[Mapping[str, Any], ...]:
    """Filter imported reports and reject incomplete requested slices."""

    selected_keys = set(selected) if selected is not None else None
    catalog = json.loads(INTERVAL_CATALOG_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        (shape_key, str(row["interval"]))
        for row in catalog["include"]
        for shape_key in shape_keys_for_families({str(row["family"])})
        if (selected_keys is None or shape_key in selected_keys)
        and (structural_interval is None or str(row["interval"]) == structural_interval)
    }
    if not expected_keys:
        raise RuntimeError("no catalog reports selected")
    filtered = tuple(
        report
        for report in reports
        if (selected_keys is None or report["shape_key"] in selected_keys)
        and (structural_interval is None or report["structural_interval"] == structural_interval)
    )
    if not filtered:
        raise RuntimeError("no reports selected")

    actual_keys = {_report_key(report) for report in filtered}
    if len(actual_keys) != len(filtered):
        raise RuntimeError("duplicate report in selected slice")
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise RuntimeError(f"missing report: {_display_key(missing[0])}")
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected report: {_display_key(unexpected[0])}")
    return filtered


def write_reports(
    reports: Collection[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write full inventories as temporary diagnostic artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for report in sorted(reports, key=_report_key):
        path = output_dir / report_filename(report)
        path.write_text(render_report(report), encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def run_inventory(
    *,
    output_dir: Path,
    fingerprint_path: Path,
    structural_interval: str,
    selected: Collection[str] | None = None,
    check: bool,
) -> InventoryRun:
    """Capture once, preserve full reports, then check or update fingerprints."""

    captured_paths: dict[tuple[str, str], Path] = {}

    def preserve_report(report: Mapping[str, Any]) -> None:
        captured_paths[_report_key(report)] = write_reports([report], output_dir)[0]

    reports = capture_all(
        structural_interval=structural_interval,
        selected=selected,
        on_report=preserve_report,
    )
    for report in reports:
        key = _report_key(report)
        if key not in captured_paths:
            preserve_report(report)
    paths = tuple(captured_paths[_report_key(report)] for report in reports)
    if check:
        mismatches = compare_fingerprints(fingerprint_path, reports)
    else:
        update_fingerprint_manifest(fingerprint_path, reports)
        mismatches = ()
    return InventoryRun(paths=paths, mismatches=mismatches)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval")
    parser.add_argument("--shape", action="append", choices=shape_keys())
    parser.add_argument("--family", action="append", choices=sorted(_FAMILY_SHAPES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINT_PATH)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-shapes", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list_shapes:
        for key in shape_keys():
            print(key)
        return 0
    selected = set(args.shape or ())
    if args.family:
        selected.update(shape_keys_for_families(args.family))
    selected_or_none = selected or None
    if args.reports_dir:
        reports = select_reports(
            load_reports(args.reports_dir),
            selected=selected_or_none,
            structural_interval=args.interval,
        )
        if args.check:
            mismatches = compare_fingerprints(
                args.fingerprints,
                reports,
                require_all=selected_or_none is None and args.interval is None,
            )
        else:
            update_fingerprint_manifest(args.fingerprints, reports)
            mismatches = ()
        for mismatch in mismatches:
            print(mismatch)
        return 1 if mismatches else 0

    run = run_inventory(
        output_dir=args.output_dir,
        fingerprint_path=args.fingerprints,
        structural_interval=args.interval or "latest",
        selected=selected_or_none,
        check=args.check,
    )
    for path in run.paths:
        print(path)
    for mismatch in run.mismatches:
        print(mismatch)
    return 1 if run.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
