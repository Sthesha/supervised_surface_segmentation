# isiZulu Grammar Parsing and Linearization Pipeline

This project provides a complete end-to-end pipeline for **processing isiZulu sentences**, parsing them into abstract syntax trees using a **GF-based grammar**, performing **tree transformations** (augmentation), and **linearizing** the resulting trees with multiple grammars for downstream NLP tasks.

---

## Project Structure

```
parsing/
│
├── corpora/                        # Original JSON corpora (e.g. grouped_cleaned_output.json)
├── data_folder/                   # Intermediate CSVs and final outputs (valid/invalid segmentations)
├── grammars/                      # PGF grammar files (ZMLargeExtChunk*.pgf)
├── parses/
│   └── various_zulu_corpora_folder/  # Stores batches, augmented, linearized, and consolidated outputs
│
├── python_scripts/                # All Python scripts
│   ├── batch_json_sentences.py
│   ├── parallel_parser.py
│   ├── consolidate_parses.py
│   ├── linguistic_tree_augmenter.py
│   ├── linearise_with_grammars.py
│   ├── validate_segmentations.py
│   └── get_sub_trees.py
│
├── bash_scripts/                  # Bash automation scripts
│   ├── parse_json.sh              # Parsing pipeline
│   └── data_generation.sh         # Tree augmentation + linearization pipeline
│
└── logs/                          # Log files created during tree linearization
```

---

## ⚙️ Step-by-Step Overview

---

### Step 1: **Prepare Sentence Batches from JSON**

Input: `grouped_cleaned_output.json`

```bash
python3 python_scripts/batch_json_sentences.py \
  corpora/grouped_cleaned_output.json \
  parses/various_zulu_corpora_folder \
  -s 5 \
  -n 100
```

- Extracts up to 100 sentences per category
- Writes them into `.txt` files
- Batches the sentences into groups of 5

---

### Step 2: **Parse the Batches to Generate Trees**

Run parsing inside Docker using the grammar:

```bash
bash bash_scripts/parse_json.sh
```

This runs:
- `parallel_parser.py` on each batch
- Saves outputs in: `parses/various_zulu_corpora_folder/batches/*.json`

---

### Step 3: **Consolidate Parsed Batches**

```bash
python3 python_scripts/combine_batches.py \
  grouped_cleaned_output \
  parses/various_zulu_corpora_folder/batches \
  parses/various_zulu_corpora_folder
```

- Combines multiple `batch_*.json` files into a single JSON file:
  ```
  parses/various_zulu_corpora_folder/grouped_cleaned_output.json
  ```

---

### Step 4: **Extract Subtrees and Trees to Augment**

```bash
bash bash_scripts/data_generation.sh
```

This pipeline performs:
1. **Sub-tree Extraction**: From `sentences_and_trees.csv`
2. **Tree Augmentation**: Generates morpheme-level variations
3. **Tree Linearization**: Linearizes all augmented trees using multiple grammars
4. **Validation**: Verifies valid vs invalid segmentations

---

## 🔍 Scripts Summary

| Script | Description |
|--------|-------------|
| `batch_json_sentences.py` | Reads JSON and splits sentences into batches |
| `parallel_parser.py` | Parses batch `.txt` files into GF abstract syntax trees |
| `combine_batches.py` | Merges `batch_*.json` into a single JSON |
| `get_sub_trees.py` | Extracts a specific column of trees for augmentation |
| `linguistic_tree_augmenter.py` | Generates variations of trees by changing polarity, tense, nouns, etc. |
| `linearise_with_grammars.py` | Linearizes trees using 1–4 grammars in parallel |
| `validate_segmentations.py` | Filters invalid segmentations from linearized results |

---

## ⚡ Example Output

- `augmented_trees.txt`: Raw GF trees (828+)
- `augmented_and_linearised.csv`: Linearized surface forms
- `valid_augmented_and_linearised.csv`: Valid segmentations
- `invalid_augmented_and_linearised.csv`: Invalid segmentations (for analysis)
- Logs in `logs/tree_processing_*.log`

---

## 🐳 Docker Container

Make sure all scripts are run inside the `parallel_parsing` Docker container:

```bash
docker exec -it parallel_parsing bash
```

To copy output back:

```bash
docker cp parallel_parsing:/root/cnlp/parsing/parsing/parses/various_zulu_corpora_folder .
```

---

## 🚀 System Requirements

- At least **32GB RAM** for larger batches
- 50+ cores for high-performance tree linearization
- [GF (Grammatical Framework)](https://www.grammaticalframework.org/) installed and compiled grammars
- Python 3.8+
- Docker (for reproducibility)

---

## 📌 Tips

- Adjust `CHUNK_SIZE`, `TIMEOUT`, `NUM_CORES` for performance tuning
- Always validate augmented results using `validate_segmentations.py`
- Clean logs regularly if debugging is disabled

---

##  Maintainer

**Sthembiso Mkhwanazi**  
Researcher in NLP for low-resource African languages  
