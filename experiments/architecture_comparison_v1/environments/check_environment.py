import sys

print("Python version:", sys.version, flush=True)

packages = ["torch", "torchvision", "timm", "transformers", "mamba_ssm", "einops"]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{pkg}: INSTALLED (v{ver})", flush=True)
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})", flush=True)
