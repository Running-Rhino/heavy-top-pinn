"""Run this file with the Python interpreter used for the coursework."""

import importlib
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])

for name in ("torch", "numpy", "scipy", "matplotlib"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {module.__version__}")
        if name == "torch":
            available = module.cuda.is_available()
            print("CUDA available:", available)
            print("Device:", module.cuda.get_device_name(0) if available else "CPU")
    except ImportError as exc:
        print(f"{name}: import failed ({exc})")
