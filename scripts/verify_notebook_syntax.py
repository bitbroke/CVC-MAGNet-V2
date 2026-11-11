import json
import ast
from pathlib import Path

nb_path = Path(r"C:\Ankshit\CVC-MAGNet-V2\CVC_MAGNet_V2_Complete_Master_Pipeline_and_Results.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Checking {len(nb['cells'])} cells in {nb_path.name}...")
errors = 0
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        clean_lines = [l for l in source.split("\n") if not l.strip().startswith("!")]
        clean_src = "\n".join(clean_lines)
        try:
            ast.parse(clean_src)
            print(f"  Cell {idx:02d} [code]: SYNTAX VALID ✅")
        except SyntaxError as e:
            print(f"  Cell {idx:02d} [code]: SYNTAX ERROR ❌: {e}")
            errors += 1

if errors == 0:
    print("\n🎉 ALL CELLS ARE 100% SYNTACTICALLY VALID AND ERROR-FREE!")
else:
    print(f"\n❌ Found {errors} syntax errors!")
