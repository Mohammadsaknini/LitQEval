# LitQEval: Literature Query Evaluation Framework

LitQEval is a framework designed to evaluate the quality of literature search queries, particularly for automatically generated queries. It includes both a comprehensive evaluation pipeline with novel metrics and a curated dataset spanning multiple research fields.

## Overview

LitQEval addresses the challenge of objectively assessing literature search query quality, especially in scenarios where queries are generated automatically through tools. The framework introduces semantic precision metrics that go beyond traditional recall-based evaluation methods.

## Key Features

- **Diverse Dataset**: Contains core publications across 21 research topics, curated from:
  - 14 Bibliometric Analysis topics
  - 7 Systematic Literature Review topics
  
- **Novel Evaluation Metrics**:
  - Semantic precision using embedding spaces
  - Relevant publication identification
  - Query broadness penalization

## Dataset

The dataset includes:
- Core publications for each research topic
- Publication data (titles, abstracts)
- Original search queries (for SLR topics)
- Chroma vector stores containing the embeddings

## Usage

The framework can be used to:
- Evaluate literature search query quality
- Benchmark query generation tools
- Train and test automated query generation systems
- Research new evaluation metrics for literature search

## Technical Details
- Uses OpenAI's embedding models for semantic analysis
- Integration with Dimensions.ai for publication retrieval
## Project Structure
- **data/**
  - `core_publications.csv` - Core publications for each research topic
  - `metadata.xlsx` - Metadata for each core publication
  - `publications.bib` - Bibtex file for core publications (It is a JabRef file format, incase you want to have a look at the topics and their surveys)
  - `queries.json` - Contains all the used queries in order to generate the dataset (Baseline, Predicted, SLRs) 
  - **vs/** - Directory that should contain the vector stores of the dataset, please download from: 
  - **text/** - Core publications and dataset files
      - **[Topic]/** - Directory for each research topic
      - `baseline_pubs.csv` - Contains the results of the baseline query for the topic
      - `predicted_pubs.csv` - Contains the results of the predicted query for the topic
      - `slr_pubs.csv` - Contains the results of the SLR query for the topic (Not always available)
- **litqeval/**
  - **notebooks/** - Jupyter notebooks for analysis
      - `dataset_eda.ipynb` - Dataset exploration
      - `ablation.ipynb` - Ablation study to understand the effect of different metrics
      - `empirical_thresholds.ipynb` - Empirical thresholds for the cosine similarity metric
      - ...
  - `eval_utils.py` - Evaluation utility functions
  - `evaluation.py` - Runs the evaluation pipline using the already existing dataset
  - `data.py` - A script to extract and preprocess data based on the file: `data/publications.bib` (Requires dimensions.ai API key)
## Installation

### Prerequisites
- Python 3.10+
- Poetry (optional but recommended)
- Dimensions.ai API key
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/LitQEval.git
cd LitQEval
```

### Step 2: Install Dependencies
```bash
poetry install
```

### Step 3: Set Up Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
```

### Step 4 (Optional): Download the new dataset
1. Add a dsl.ini that conatins dimensions.ai API key. Check the dimcli repo [here](https://github.com/digital-science/dimcli) for more information.
2. run `data.py` to download the new dataset

### Step 5: Run the Evaluation Pipeline
```bash
poetry run python evaluate.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






