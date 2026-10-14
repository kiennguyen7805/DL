"""Resolve historical Day 2 evidence paths without rewriting frozen manifests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def resolve_result_reference(root: Path, value: str) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe result path: {value}")
    parts = relative.parts
    if parts[:2] == ("results", "day2"):
        parts = ("results", "lam_sach_du_lieu", *parts[2:])
    project = root.resolve()
    result = project.joinpath(*parts).resolve()
    if not result.is_relative_to(project):
        raise ValueError(f"Result path outside project: {value}")
    return result
