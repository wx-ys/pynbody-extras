from __future__ import annotations

from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension


ROOT = Path(__file__).parent


# Optional Rust extension providing high-performance implementations
# for tree and gravity operations. When the Rust toolchain is not
# available, installation will fall back to the pure-Python package
# without this extension.
rust_extensions = [
    RustExtension(
        "pynbodyext._rust",
        path=str(ROOT / "crates" / "pynbodyext-rust" / "Cargo.toml"),
        binding=Binding.PyO3,
        optional=True,
    ),
]


if __name__ == "__main__":
    setup(rust_extensions=rust_extensions)
