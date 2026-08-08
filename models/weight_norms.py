"""Load and validate target weight norms for model initialization."""

import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "model.")
_ARCHITECTURE_FIELDS = ("vocab_size", "n_layer", "n_head", "n_embd")


def canonical_parameter_name(name):
    """Remove wrappers added by DDP, torch.compile, or model containers."""
    name = name.strip()

    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True

    return name


def validate_target_norms(target_norms):
    """Return a canonical ``parameter_name -> positive finite norm`` mapping."""
    if not isinstance(target_norms, Mapping):
        raise TypeError("target_norms must be a JSON object")

    validated = {}
    for raw_name, raw_norm in target_norms.items():
        if not isinstance(raw_name, str):
            raise TypeError("Target-norm parameter names must be strings")

        name = canonical_parameter_name(raw_name)
        if not name:
            raise ValueError("Found an empty target-norm parameter name")

        if isinstance(raw_norm, bool) or not isinstance(raw_norm, Real):
            raise TypeError(f"Target norm for {name} must be a JSON number")

        norm = float(raw_norm)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"Target norm for {name} must be positive and finite; got {raw_norm}"
            )

        if name in validated:
            raise ValueError(
                f"Duplicate target norm after canonicalizing parameter name {name}"
            )

        validated[name] = norm

    if not validated:
        raise ValueError("Target-norm JSON contains no parameters")

    return validated


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in target-norm JSON: {key}")
        result[key] = value
    return result


def _validate_architecture_metadata(metadata, model_config):
    if not isinstance(metadata, Mapping):
        raise TypeError("The JSON 'metadata' field must be an object")

    if model_config is None:
        return

    for field in _ARCHITECTURE_FIELDS:
        if field not in metadata:
            continue

        expected = getattr(model_config, field)
        actual = metadata[field]
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"metadata.{field} must be an integer")
        if actual != expected:
            raise ValueError(
                f"Target norms require {field}={actual}, but the model uses "
                f"{field}={expected}"
            )


def load_target_norms(path, model_config=None):
    """Load target norms from a JSON file and optionally check model metadata.

    The preferred schema is::

        {
          "metadata": {"n_layer": 12, "n_embd": 512},
          "target_norms": {"transformer.wte.weight": 574.3865}
        }

    A direct ``parameter_name -> norm`` JSON object is also accepted.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Target-norm file must use the .json extension: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file, object_pairs_hook=_object_without_duplicate_keys)

    if not isinstance(data, Mapping):
        raise TypeError("Target-norm JSON must contain an object at the top level")

    if "target_norms" in data:
        unknown_fields = sorted(set(data) - {"metadata", "target_norms"})
        if unknown_fields:
            raise ValueError(
                "Unknown top-level target-norm JSON fields: "
                + ", ".join(unknown_fields)
            )

        metadata = data.get("metadata", {})
        _validate_architecture_metadata(metadata, model_config)
        target_norms = data["target_norms"]
    else:
        target_norms = data

    return validate_target_norms(target_norms)
