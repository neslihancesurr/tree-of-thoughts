# Tree of Thoughts

A small Python implementation of the Tree of Thoughts (ToT) approach for exploring multiple interpretations of ambiguous natural-language sentences using local Ollama models.

This project generates several candidate interpretations for a sentence, scores them with an evaluator model, selects the most promising ones, and refines them to improve ambiguity coverage.

## Features

- Generates multiple initial interpretations of a sentence
- Uses a second model to evaluate how well each interpretation captures ambiguity
- Refines the strongest candidate interpretations
- Uses a beam-search style selection to keep the best ideas
- Works entirely with local Ollama models

## Project structure

- `TreeOfThoughts.py` — core Tree-of-Thoughts implementation
- `main.py` — example runner showing how to execute the framework on sample sentences
- `crane` — evaluation file

## Requirements

- Python 3.9+
- Ollama installed and running locally
- The following models available in Ollama:
  - `qwen2.5:0.5b`
  - `gemma3:latest`

## Setup

1. Install Ollama: https://ollama.com/download
2. Pull the required models:

```bash
ollama pull qwen2.5:0.5b
ollama pull gemma3:latest
```

3. Install the Python package used by the project:

```bash
pip install ollama
```

## Run the example

From the repository root:

```bash
python main.py
```

The script evaluates a few ambiguous sentences and prints the generated interpretations plus score statistics.

## Example usage

The main script initializes the models like this:

```python
from TreeOfThoughts import TreeOfThoughts

tot = TreeOfThoughts(
    generator_model="qwen2.5:0.5b",
    evaluator_model="gemma3:latest"
)

results = tot.run_tot(
    sentence="I saw the man with the telescope.",
    initial_k=5,
    refinement_k=2,
    beam_width=3
)
```

## Notes

- This project is intended for experimentation and research around ambiguity handling in language models.
- Model performance depends on the quality and speed of the local Ollama installation.
- The evaluation prompt expects a JSON response with a numeric score and a justification.
