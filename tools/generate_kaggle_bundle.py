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
    notebook_cells = []
    
    # 1. README / Header
    readme_content = read_file("README.md")
    notebook_cells.append(create_notebook_cell(
        "# LEMA: Layer-wise Efficient Memory Abstraction\n"
        "This notebook is a self-contained LEMA workspace. It includes the library, examples, tasks, and documentation.\n\n"
        "--- \n" + readme_content,
        cell_type="markdown"
    ))
    
    # 2. Workspace Setup
    notebook_cells.append(create_notebook_cell(
        "!mkdir -p src/lema/core src/lema/engine src/lema/models src/lema/utils tasks examples/kaggle docs output",
        cell_type="code"
    ))
    
    # 3. Inject ALL Library Code
    lib_files = glob.glob("src/lema/**/*.py", recursive=True)
    for file_path in lib_files:
        if "__pycache__" in file_path: continue
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))
        
    # 4. Inject ALL Tasks
    task_files = glob.glob("tasks/*.py")
    for file_path in task_files:
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 5. Inject ALL Examples
    example_files = glob.glob("examples/*.py") + glob.glob("examples/kaggle/*.py")
    for file_path in example_files:
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 6. Inject Documentation
    doc_files = glob.glob("docs/*.md")
    for file_path in doc_files:
        content = read_file(file_path)
        source = f"### {file_path}\n\n{content}"
        notebook_cells.append(create_notebook_cell(source, cell_type="markdown")
        )
        
    # 7. Environment Setup
    notebook_cells.append(create_notebook_cell(
        "import sys, os\n"
        "sys.path.append(os.path.abspath('src'))\n"
        "!pip install -q safetensors accelerate peft transformers\n"
        "print('LEMA Environment Fully Loaded.')"
    ))
    
    # 8. EXECUTABLE: Speed Benchmark
    notebook_cells.append(create_notebook_cell("# RUN SPEED BENCHMARK", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run examples/kaggle/speed_benchmark.py"))

    # 9. EXECUTABLE: SmolLM Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN SMOL-LM FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_smollm.py"))

    # 10. EXECUTABLE: Llama-7B Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN LLAMA-7B FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_llama_7b.py"))

    # Construct Notebook JSON
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = "examples/kaggle/benchmark.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Successfully generated comprehensive bundle {output_path} with {len(notebook_cells)} cells.")

if __name__ == "__main__":
    main()