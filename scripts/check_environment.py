from __future__ import annotations

import ctypes.util
import importlib
from importlib import metadata
import platform
import sys


PACKAGES = ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "PyYAML", "h5py")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")

    for library in ("hdf5_serial", "hdf5"):
        print(f"lib{library}: {ctypes.util.find_library(library) or 'not found'}")

    failures = []
    for module_name in ("torch", "ultralytics", "yaml", "PIL"):
        try:
            module = importlib.import_module(module_name)
            print(f"import {module_name}: OK")
            if module_name == "torch":
                print(f"CUDA available: {module.cuda.is_available()}")
        except Exception as error:
            failures.append(module_name)
            print(f"import {module_name}: FAILED ({type(error).__name__}: {error})")

    if failures:
        print("Environment is not ready for model training.")
        return 1
    print("Core Python imports are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
