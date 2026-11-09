"""Explicit descriptors of caller-owned local files; serialization never does IO."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

__all__ = ["ExternalFileRef", "ExternalFileInspection"]

_SCHEMA = "pynbodyext.result.external-file-ref/v1"
_FIELDS = {"schema", "uri", "media_type", "size_bytes", "sha256"}


def _local_path(uri: str) -> Path:
    if not isinstance(uri, str) or not uri.strip() or "\0" in uri:
        raise ValueError("uri must be a nonempty local absolute path or file URI")
    if os.path.isabs(uri):
        return Path(uri)
    # urlsplit strips leading controls/spaces and embedded tabs/newlines. Reject
    # those URI spellings rather than silently inspecting a different location.
    if ord(uri[0]) <= 32 or any(character in uri for character in "\r\n\t"):
        raise ValueError("file URI must not contain ignored whitespace or control characters")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc.lower() not in ("", "localhost")
        or parsed.query
        or parsed.fragment
        or "?" in uri
        or "#" in uri
        or re.search(r"%(?![0-9a-fA-F]{2})", parsed.path)
    ):
        raise ValueError("uri must be a local absolute path or file URI without query or fragment")
    try:
        path = unquote(parsed.path, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("file URI must contain valid UTF-8 percent escapes") from exc
    if not os.path.isabs(path) or "\0" in path:
        raise ValueError("file URI must describe an absolute local path")
    return Path(path)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate external-file descriptor field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ExternalFileInspection:
    """One explicit observation; available alone does not establish content equality.

    ``checksum_verified`` is true only when a requested SHA256 comparison matched.
    ``reason`` distinguishes missing files, size/type changes, skipped checks,
    absent checksums and checksum matches/mismatches. No observation locks a file.
    """

    status: Literal["available", "missing", "changed"]
    reason: str
    checksum_verified: bool = False


@dataclass(frozen=True, slots=True)
class ExternalFileRef:
    """An immutable descriptor of a caller-owned local file, without implicit IO.

    Parameters
    ----------
    uri : str
        Absolute local path or absolute file URI with empty/localhost authority.
        Original text is preserved; no existence check or symlink resolution occurs.
    media_type : str
        Nonempty caller-supplied media type; contents are not sniffed.
    size_bytes : int or None, default None
        Expected nonnegative size. Booleans and non-integers are rejected.
    sha256 : str or None, default None
        Expected lowercase 64-hex checksum, checked only by explicit inspection.

    Notes
    -----
    A pinned ResultStore version preserves this descriptor, not the external bytes.
    Fetching it cannot restore an overwritten file. Callers own copying, versioned
    filenames and deletion. Equality and codec content hashes describe metadata only.
    """

    uri: str
    media_type: str
    size_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _local_path(self.uri)
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("media_type must be a nonempty string")
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes < 0):
            raise ValueError("size_bytes must be a nonnegative integer or None")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters or None")

    def _to_payload(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "uri": self.uri,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def _from_payload(cls, data: Any) -> ExternalFileRef:
        if not isinstance(data, dict) or set(data) != _FIELDS or data["schema"] != _SCHEMA:
            raise ValueError("unsupported external-file descriptor schema or fields")
        return cls(data["uri"], data["media_type"], data["size_bytes"], data["sha256"])

    def to_json(self) -> str:
        """Serialize all descriptor fields and schema without accessing the file."""
        return json.dumps(self._to_payload(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> ExternalFileRef:
        """Decode strict versioned JSON without IO; invalid fields raise ValueError."""
        return cls._from_payload(json.loads(text, object_pairs_hook=_unique_object))

    def inspect(self, verify_checksum: bool = False) -> ExternalFileInspection:  # noqa: PLR0911
        """Observe the local file, optionally comparing a caller-supplied SHA256.

        Parameters
        ----------
        verify_checksum : bool, default False
            Read binary blocks and compare SHA256 only if a digest is supplied.
            Without one, availability never claims equality of file contents.

        Returns
        -------
        ExternalFileInspection
            Status available/missing/changed and reason. Reasons are not_found,
            not_regular_file, size_mismatch, checksum_not_checked,
            checksum_not_supplied, checksum_mismatch and checksum_match.
            checksum_verified is true only for checksum_match.

        Notes
        -----
        Default availability is a stat observation, not proof of readability.
        Symlinks are followed; directories are changed and broken links missing.
        PermissionError and other access errors propagate. Concurrent external
        writes are not locked; this observation does not create a file snapshot.
        No file is copied, created, modified or deleted by inspection.
        """
        if type(verify_checksum) is not bool:
            raise ValueError("verify_checksum must be a bool")
        path = _local_path(self.uri)
        try:
            facts = path.stat()
            if not stat.S_ISREG(facts.st_mode):
                return ExternalFileInspection("changed", "not_regular_file")
            if self.size_bytes is not None and facts.st_size != self.size_bytes:
                return ExternalFileInspection("changed", "size_mismatch")
            if not verify_checksum:
                return ExternalFileInspection("available", "checksum_not_checked")
            if self.sha256 is None:
                return ExternalFileInspection("available", "checksum_not_supplied")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != self.sha256:
                return ExternalFileInspection("changed", "checksum_mismatch")
            return ExternalFileInspection("available", "checksum_match", checksum_verified=True)
        except FileNotFoundError:
            return ExternalFileInspection("missing", "not_found")
