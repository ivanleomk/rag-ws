# Introduction

This is a sample repository to show you how to progressively build out a more complex RAG application.

Here are the steps to start

1. Create Virtual Environment - We recommend [uv](https://github.com/astral-sh/uv) which has good support for virtual environments.

```bash
uv venv venv
```

2. Install the necessary dependencies

```bash
uv pip install -r requirements.txt
```

3. Run setup.py so that you can get the requirements configured nicely

```bash
python3 ./notebooks/setup.py
```

4. Add your new kernel to the jupyerlab runtime

```bash
python -m ipykernel install --user --name=rag_workshop --display-name="Rag Workshop"
```

5. Start Jupyter Lab and start working

```bash
jupyter notebook
```
