"""Namespace hygiene: wrapper instance attributes must never collide with a
wrapped provider client's own attribute namespace (e.g. openai-python's
private `_client`). Everything the wrapper stores on itself lives under
`_solwyn_*`; all other names belong to the wrapped client.
"""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "solwyn"
_WRAPPER_CLASSES = {
    "_SolwynBase": "_base.py",
    "Solwyn": "client.py",
    "AsyncSolwyn": "client.py",
}


def _self_attribute_stores(class_node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(class_node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    return names


@pytest.mark.unit
def test_wrapper_instance_attributes_use_solwyn_namespace() -> None:
    offenders: list[str] = []
    for cls_name, filename in _WRAPPER_CLASSES.items():
        tree = ast.parse((_SRC / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                offenders += [
                    f"{cls_name}.{attr}"
                    for attr in _self_attribute_stores(node)
                    if not attr.startswith("_solwyn_")
                ]
    assert offenders == [], f"non-namespaced wrapper attrs: {sorted(offenders)}"
