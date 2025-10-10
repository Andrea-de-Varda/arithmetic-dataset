# Arithmetic Problem Dataset Generator

A flexible tool for generating arithmetic datasets with precise control over difficulty and format. Ideal for training and evaluating language models on mathematical reasoning.

## Features

**Difficulty Control:**
- Number of terms (2-term: `32 + 41` vs 3-term: `32 + 41 - 25`)
- Number of digits (2-digit: `32` vs 3-digit: `320`)
- Carry/borrow operations (controllable percentage)
- Solution digit filtering

**Multiple Formats (5 total):**
- Numeric: `32 + 41 =`
- English: `thirty-two plus forty-one equals`
- Spanish: `treinta y dos más cuarenta y uno es igual a`
- Italian: `trentadue più quarantuno fa`
- Embedded: `Alex has 32 apples. Emma has 41 apples. How many apples do they have together?`

**Output Formats:**
- JSON (default): Standard JSON arrays
- HELM: JSONL format compatible with Stanford's HELM benchmark

**Dataset Splits:** 80% train, 10% validation, 10% test (stratified to preserve statistics)

## Quick Start

```bash
pip install -r requirements.txt
python generate_arithmetic_data.py -n 1000
```

This creates 5 files with matched IDs:
- `arith_dataset_numeric.json`
- `arith_dataset_english.json`
- `arith_dataset_spanish.json`
- `arith_dataset_italian.json`
- `arith_dataset_embedded.json`

## Usage Examples

### Basic Generation
```bash
# Default settings (50-50 split for all parameters)
python generate_arithmetic_data.py -n 1000

# HELM format
python generate_arithmetic_data.py -n 1000 --output-format helm
```

### Control Difficulty
```bash
# Mostly 3-term problems with 3-digit numbers
python generate_arithmetic_data.py -n 1000 \
  --terms 0.3 0.7 \
  --digits 0.2 0.8 \
  --carry-percentage 0.7

# Only 2-digit solutions
python generate_arithmetic_data.py -n 1000 --solution-digits 2

# Higher carry/borrow percentage
python generate_arithmetic_data.py -n 1000 --carry-percentage 0.8
```

### Optional Constraints
```bash
python generate_arithmetic_data.py -n 1000 \
  --avoid-repeated-digits \
  --avoid-clean-multiples \
  --avoid-reverse-pairs
```

### Reproducible Generation
```bash
python generate_arithmetic_data.py -n 1000 --seed 42
```

## Command-Line Arguments

**Dataset Size:**
- `-n, --num-problems`: Number of problems to generate (default: 1000)

**Difficulty:**
- `--terms PROB_2 PROB_3`: Distribution for 2-term vs 3-term (default: 0.5 0.5)
- `--digits PROB_2 PROB_3`: Distribution for 2-digit vs 3-digit (default: 0.5 0.5)
- `--solution-digits N`: Filter to keep only N-digit solutions
- `--carry-percentage P`: Proportion with carry/borrow (default: 0.5)

**Constraints:**
- `--avoid-repeated-digits`: Avoid numbers like 121, 77, 88
- `--avoid-clean-multiples`: Avoid multiples of 10 (70, 80, 100)
- `--avoid-reverse-pairs`: For 2-term addition, avoid pairs like 34+21 and 21+34

**Output:**
- `-o, --output-prefix`: Output filename prefix (default: arith_dataset)
- `--output-format`: `json` (default) or `helm` (JSONL format)
- `--seed`: Random seed for reproducibility

## Output Format

### JSON Format (Default)

Each file contains an array of problems:

```json
{
  "id": "prob_000123",
  "prompt": "32 + 41 =",
  "answer": " 73",
  "ground_truth": 73,
  "split": "train",
  "has_carry": false,
  "n_terms": 2,
  "n_digits": 2
}
```

### HELM Format (JSONL)

Each line is a JSON object (Stanford HELM benchmark compatible):

```json
{
  "input": {
    "text": "32 + 41 ="
  },
  "references": [
    {
      "output": {
        "text": " 73"
      },
      "tags": ["correct"]
    }
  ],
  "split": "train",
  "id": "prob_000123"
}
```

### Reading Files

**Python (JSON):**
```python
import json

with open('arith_dataset_numeric.json', 'r') as f:
    data = json.load(f)

for problem in data:
    print(f"{problem['prompt']} -> {problem['answer'].strip()}")
```

**Python (HELM/JSONL):**
```python
import json

problems = []
with open('arith_dataset_numeric.jsonl', 'r') as f:
    for line in f:
        problems.append(json.loads(line))

for p in problems:
    print(f"{p['input']['text']} -> {p['references'][0]['output']['text'].strip()}")
```

## Dataset Statistics

The included dataset (10,000 problems, seed 42):
- Train: 7,996 (80%), Val: 997 (10%), Test: 1,007 (10%)
- 2-term: 48.3%, 3-term: 51.7%
- 2-digit: 48.8%, 3-digit: 51.2%
- With carry/borrow: 50.3%, Without: 49.7%

All problems are unique with stratified train/val/test splits to preserve statistical distributions.

## Embedded Format Examples

The embedded format generates contextual word problems:

- **Addition**: "Alex has 12 apples. Emma has 41 apples. How many apples do they have together?"
- **Subtraction**: "Noah had 87 coins. Noah gave 32 coins to Olivia. How many coins does Noah have left?"
- **Multi-term**: "Sophia had 50 stickers. Emma gave Sophia 30 more stickers. Then Sophia gave 25 stickers away. How many stickers does Sophia have now?"

Features 36 character names and 15 object types with deterministic story generation (same problem = same story).

## Notes

- Leading spaces in answers (e.g., `" 73"`) are included for LLM compatibility
- IDs are matched across all 5 format files for easy cross-referencing
- Carry detection works for addition, borrow detection for subtraction
- The `--avoid-reverse-pairs` constraint only applies to 2-term addition (commutative operations)
- Stratified splitting ensures train/val/test sets maintain the same statistical properties

## License

MIT License
