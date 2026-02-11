import os
import json
import glob

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_notebook_cell(source_code, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_code.splitlines(keepends=True)
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def main():
    root_dir = "src"
    notebook_cells = []
    
    # 1. Header
    notebook_cells.append(create_notebook_cell(
        "# LEMA Benchmark Notebook\n"
        "This notebook was auto-generated to verify the LEMA framework on Kaggle GPUs.\n"
        "It compares Standard Fine-Tuning (PEFT) vs LEMA (Virtual Memory) VRAM usage.",
        cell_type="markdown"
    ))
    
    # 2. Setup Library Code
    notebook_cells.append(create_notebook_cell(
        "!mkdir -p src/lema/core src/lema/engine src/lema/models src/lema/utils tasks",
        cell_type="code"
    ))
    
    # Find all python files in src and tasks
    files = glob.glob(f"{root_dir}/**/*.py", recursive=True)
    files += glob.glob("tasks/**/*.py", recursive=True)
    
    for file_path in files:
        content = read_file(file_path)
        # Generate cell to write file
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))
        
    # 3. Add Path Setup
    notebook_cells.append(create_notebook_cell(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.path.abspath('src'))\n"
        "print('LEMA library loaded.')"
    ))
    
    # 4. Install Dependencies
    notebook_cells.append(create_notebook_cell(
        "!pip install -q safetensors accelerate peft transformers"
    ))
    
    # 5. Benchmark Script (The actual logic)
    # Pointing to the new location in examples/kaggle/ 
    try:
        benchmark_code = read_file("examples/kaggle/speed_benchmark.py")
    except FileNotFoundError:
        print("Error: examples/kaggle/speed_benchmark.py not found.")
        return

    notebook_cells.append(create_notebook_cell(benchmark_code))
    
    # Construct Notebook JSON
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Save to examples/kaggle/
    output_path = "examples/kaggle/benchmark.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Generated {output_path}")

if __name__ == "__main__":
    main()
