# Mobix: Drawing-to-BIM Large Language Model

## Overview

Mobix is a conceptual framework and sample code for converting architectural drawings (PDFs, CAD, images, or text descriptions) into structured building data (BIM/JSON) using Large Language Models (LLMs) and multimodal AI.

## Features

- Converts textual descriptions of architectural drawings into structured BIM-like JSON.
- Scaffold for integrating image-to-text models (e.g., BLIP, LLaVa) for multimodal input.
- Example code for using HuggingFace Transformers and PyTorch.
- Ready for extension to full Drawing-to-BIM pipelines.

## Requirements

- Python 3.9+
- PyTorch
- transformers
- tqdm
- Pillow
- (Optional for OCR) pytesseract
- (Optional for CAD/BIM) ezdxf, ifcopenshell

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

1. **Clone or download this repository.**
2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On Linux/Mac:
   source env/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Navigate to the project directory:**
   ```bash
   cd mobix
   ```
5. **Run the notebook or Python scripts:**
   - To start the notebook:
     ```bash
     jupyter notebook app.ipynb
     ```
   - Or run your Python code as needed.

## Example Usage

```python
from mobix import DrawingToBIMLLM

llm = DrawingToBIMLLM()
drawing_text = "Ground floor plan: 2 rooms, each 4x5m, separated by a wall. Door on north wall, window on east wall."
bim_json = llm.convert_drawing_to_bim(drawing_text)
print(bim_json)
```

## Jupyter Notebook

See `app.ipynb` for a notebook example.

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Salesforce BLIP](https://github.com/Salesforce/BLIP)
- [LLaVa](https://github.com/haotian-liu/LLaVA)
- [ifcopenshell](https://ifcopenshell.org/)
