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
    notebook_cells.append(create_notebook_cell(f"%%writefile README.md\n{readme_content}"))
    
    # 2. Workspace Setup
    notebook_cells.append(create_notebook_cell(
        "!mkdir -p src/lema/core src/lema/engine src/lema/models src/lema/utils src/lema/csrc tasks examples/kaggle docs tests output",
        cell_type="code"
    ))
    
    # 3. Inject ALL Library Code
    lib_files = glob.glob("src/lema/**/*.py", recursive=True) + glob.glob("src/lema/**/*.cpp", recursive=True) + glob.glob("src/lema/**/*.h", recursive=True)
    for file_path in lib_files:
        if "__pycache__" in file_path: continue
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 4. Inject Project Configs
    for file_path in ["setup.py", "pyproject.toml", "requirements.txt"]:
        if os.path.exists(file_path):
            content = read_file(file_path)
            source = f"%%writefile {file_path}\n{content}"
            notebook_cells.append(create_notebook_cell(source))
        
    # 5. Inject ALL Tasks
    task_files = glob.glob("tasks/*.py")
    for file_path in task_files:
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 6. Inject ALL Examples
    example_files = glob.glob("examples/*.py") + glob.glob("examples/kaggle/*.py")
    for file_path in example_files:
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 7. Inject Tests (for verification)
    test_files = glob.glob("tests/*.py")
    for file_path in test_files:
        content = read_file(file_path)
        source = f"%%writefile {file_path}\n{content}"
        notebook_cells.append(create_notebook_cell(source))

    # 8. Inject Documentation
    doc_files = glob.glob("docs/*.md")
    for file_path in doc_files:
        content = read_file(file_path)
        source = f"### {file_path}\n\n{content}"
        notebook_cells.append(create_notebook_cell(source, cell_type="markdown")
        )
        
    # 9. Environment Setup
    notebook_cells.append(create_notebook_cell(
        "import sys, os, torch, subprocess\n"
        "sys.path.append(os.path.abspath('src'))\n"
        "\n"
        "# Set temporary cache directories for Kaggle to avoid filling up /kaggle/working\n"
        "os.environ['HF_HOME'] = '/tmp/huggingface_cache'\n"
        "os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'\n"
        "os.makedirs('/tmp/huggingface_cache', exist_ok=True)\n"
        "\n"
        "!pip install -q safetensors accelerate peft transformers ninja\n"
        "!pip install -q torchao --upgrade  # PEFT needs >0.16.0\n"
        "\n"
        "if torch.cuda.is_available():\n"
        "    capability = torch.cuda.get_device_capability()\n"
        "    if capability[0] < 7:\n"
        "        print(f'\\033[93mWARNING: Current GPU capability ({capability[0]}.{capability[1]}) is too low for some PyTorch kernels. ')\n"
        "        print('If you experience CUDA errors, switch to T4 (sm_75) or L4 (sm_89) in Kaggle Settings.\\033[0m')\n"
        "\n"
        "    print('Compiling C++ extensions... this may take a minute.')\n"
        "    try:\n"
        "        # Compile in-place only (no pip install — local source takes priority over stale site-packages)\n"
        "        ret = subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], capture_output=True, text=True)\n"
        "        if ret.returncode != 0:\n"
        "            print(f'Warning: C++ compilation failed.\\nSTDOUT: {ret.stdout}\\nSTDERR: {ret.stderr}')\n"
        "            print('Falling back to pure Python mode.')\n"
        "        else:\n"
        "            print('C++ Backend compiled successfully (in-place).')\n"
        "    except Exception as e:\n"
        "        print(f'Warning: Error during C++ compilation ({e}). Falling back to pure Python mode.')\n"
        "else:\n"
        "    print('CUDA not available. Skipping C++ backend compilation (Pure Python mode).')\n"
        "\n"
        "# Import from source (not stale installed package)\n"
        "sys.path.insert(0, os.path.abspath('src'))\n"
        "import lema\n"
        "from lema.core import memory\n"
        "print(f'LEMA Environment Fully Loaded. HAS_CPP_BACKEND: {memory.HAS_CPP_BACKEND}')\n"
        "if not memory.HAS_CPP_BACKEND and torch.cuda.is_available():\n"
        "    print('NOTE: Running in pure Python mode. Performance may be lower.')"
    ))
    
    # 10. EXECUTABLE: C++ vs Python Backend Benchmark (priority test)
    notebook_cells.append(create_notebook_cell(
        "# C++ BACKEND vs PURE PYTHON BENCHMARK\n"
        "# This benchmark isolates each operation to measure C++ vs Python overhead.\n"
        "# Results will show exactly which operations benefit from C++ and which don't.",
        cell_type="markdown"
    ))
    notebook_cells.append(create_notebook_cell(
        "import sys, gc, os, time, torch\n"
        "sys.path.append(os.path.abspath('src'))\n"
    ))
    notebook_cells.append(create_notebook_cell("%run examples/kaggle/cpp_vs_python_benchmark.py"))

    # 11. EXECUTABLE: Scaling Benchmark (real fine-tuning at scale)
    notebook_cells.append(create_notebook_cell(
        "# FINE-TUNING SCALING BENCHMARK\n"
        "# Tests PEFT vs LEMA across batch sizes and sequence lengths.\n"
        "# Shows where PEFT OOMs and LEMA keeps training.",
        cell_type="markdown"
    ))
    notebook_cells.append(create_notebook_cell(
        "import sys, gc, os, time, torch\n"
        "sys.path.append(os.path.abspath('src'))\n"
    ))
    notebook_cells.append(create_notebook_cell(
        "%run examples/kaggle/scaling_benchmark.py\n"
        "# WARNING: This benchmark takes a while. It runs TinyLlama 1.1B\n"
        "# across multiple batch sizes (1,2,4,8) and seq lengths (128,256,512)."
    ))

    # 13. EXECUTABLE: Verify Mistral/Mixtral Adapters
    notebook_cells.append(create_notebook_cell("# VERIFY MISTRAL/MIXTRAL ADAPTERS", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("!pytest tests/test_mistral_mixtral.py"))

    # 14. EXECUTABLE: Speed Benchmark
    notebook_cells.append(create_notebook_cell("# RUN SPEED BENCHMARK", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run examples/kaggle/speed_benchmark.py"))

    # 15. EXECUTABLE: SmolLM Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN SMOL-LM FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_smollm.py"))

    # 16. EXECUTABLE: Llama-7B Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN LLAMA-7B FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_llama_7b.py"))

    # 17. EXECUTABLE: Mistral Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN MISTRAL FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_mistral.py"))

    # 18. EXECUTABLE: Mixtral Fine-tuning
    notebook_cells.append(create_notebook_cell("# RUN MIXTRAL FINE-TUNING", cell_type="markdown"))
    notebook_cells.append(create_notebook_cell("%run tasks/fine_tune_mixtral.py"))

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