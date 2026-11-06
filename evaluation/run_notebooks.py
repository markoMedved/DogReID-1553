#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell

def run_notebook(nb_path: Path):
    print(f"=== Executing {nb_path.name} ===")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            print(f"-> Running cell {i}...")
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            redirected_output = StringIO()
            sys.stdout = redirected_output
            sys.stderr = redirected_output
            
            try:
                result = shell.run_cell(source)
                cell_out = redirected_output.getvalue()
                outputs = []
                if cell_out:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": cell_out.splitlines(keepends=True)
                    })
                if result.error_in_exec:
                    print(f"[ERROR] In cell {i}: {result.error_in_exec}", file=old_stderr)
                    outputs.append({
                        "ename": type(result.error_in_exec).__name__,
                        "evalue": str(result.error_in_exec),
                        "output_type": "error",
                        "traceback": [str(result.error_in_exec)]
                    })
                cell["outputs"] = outputs
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"=== Successfully executed and saved {nb_path.name} ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--closed", action="store_true", help="Run evaluate_closed_world.ipynb")
    parser.add_argument("--open", action="store_true", help="Run evaluate_open_world.ipynb")
    parser.add_argument("--ablations", action="store_true", help="Run evaluate_ablations.ipynb")
    parser.add_argument("--identity", action="store_true", help="Run evaluate_identity_and_confusion.ipynb")
    parser.add_argument("--all", action="store_true", help="Run all 4 notebooks")
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent

    if args.closed or args.all:
        run_notebook(eval_dir / "evaluate_closed_world.ipynb")
    if args.open or args.all:
        run_notebook(eval_dir / "evaluate_open_world.ipynb")
    if args.ablations or args.all:
        run_notebook(eval_dir / "evaluate_ablations.ipynb")
    if args.identity or args.all:
        nb_id = eval_dir / "failure_case" / "evaluate_identity_and_confusion.ipynb"
        if not nb_id.exists():
            nb_id = eval_dir / "evaluate_identity_and_confusion.ipynb"
        run_notebook(nb_id)
