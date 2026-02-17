# Phytomni Manuscript Code

Phytomni: Reproducibility code and notebooks for "An agentic AI for scientific discovery and design in plant research"

## Overview

This repository provides the code, scripts, and notebooks to reproduce figures, tables, and quantitative results reported in the manuscript "Phytomni: An agentic AI for scientific discovery and design in plant research." Phytomni is a domain-specific, LLM-powered multi-agent system built on the Model Context Protocol (MCP) that integrates a plant-focused full-text knowledge base (~4.0M publications plus abstracts and patents), multi-omics data spanning 65 species, and 125 bioinformatics tools. The platform orchestrates hierarchically coordinated agents—Knowledge, Data, Analyst—and composite agents (e.g., In Silico Research Agent, Deep Genome Agent, Gene Network Agent, Digital Design Agent) to automate literature-grounded reasoning, data retrieval, and end-to-end bioinformatic analyses.

Code in this repository is organized by figure directory and has been tested with Python 3.8+ and R 4.0+. Executing the provided Python/R scripts and Jupyter notebooks reproduces the main and supplementary figures, including: (i) Knowledge Agent benchmarks; (ii) Data Agent natural-language-to-SQL performance on plant multi-omics; (iii) Analyst Agent goal-completion across diverse bioinformatics workflows; (iv) In Silico Research Agent paper-replication efficiency; and (v) Deep Genome Agent functional summarization and confabulation analyses.

## Development Environment

### Python Dependencies
```bash
pip install matplotlib pandas seaborn plotly numpy openpyxl
```

### R Dependencies
```bash
R -e "install.packages(c('tidyverse', 'scales', 'treemapify'))"
```

### Jupyter Notebook Support
```bash
pip install jupyterlab
# For R kernel in Jupyter (optional)
R -e "install.packages(c('IRkernel'))"
R -e "IRkernel::installspec()"
```

## Repository Structure

```
Phytomni-Manuscript/
├── Fig. 2/                              # Main figures (Plotly interactive plots)
├── Fig. 3/                              # Paper replication analysis
├── Extended Data Fig. 5/                # Extended data figures
├── Extended Data Fig. 6/                # Extended data figures
├── Supplementary Fig. 6/                # Supplementary figures (Plotly)
├── Supplementary Fig. 7-9/              # PhytoBench-Gene analysis
└── Supplementary Fig. 13/               # Supplementary figures
```

### Directory Details

| Directory | Content | Type |
|-----------|---------|------|
| `Fig. 2/` | Model performance comparisons | Jupyter Notebook |
| `Fig. 3/` | Paper replication benchmark | Jupyter Notebook + Excel |
| `Extended Data Fig. 5/` | Publication trends, document types | R Script + Jupyter Notebook |
| `Extended Data Fig. 6/` | Accuracy heatmaps, model comparisons | Python Script + Jupyter Notebook |
| `Supplementary Fig. 6/` | Interactive visualizations | Jupyter Notebook |
| `Supplementary Fig. 7-9/` | PhytoBench-Gene analysis | Jupyter Notebook + TSV Data |
| `Supplementary Fig. 13/` | Additional analyses | Jupyter Notebook |

## Data

Data files are located within their respective figure directories:

- **Fig. 3/**: `PhytoBench-Paper-for_plot.xlsx` — Paper replication benchmark data
- **Supplementary Fig. 7-9/PhytoBench-Gene-for_plot/**: Gene functional summarization scores
  - Overall scores: `score.tsv`, `score.uncharacterized.tsv`, `score.well_studied.tsv`
  - Species-specific scores: `score.arabidopsis.tsv`, `score.maize.tsv`, `score.rice.tsv`, `score.soybean.tsv`, `score.wheat.tsv`
  - Categorized species scores: `score.uncharacterized.*.tsv`, `score.well_studied.*.tsv`

## Running the Code

### Jupyter Notebooks
```bash
jupyter lab
# Then open and run cells in the desired notebook:
# - "Fig. 2/fig. 2.ipynb"
# - "Fig. 3/fig. 3.ipynb"
# - "Extended Data Fig. 5/extended_data_fig. 5d.ipynb"
# - "Extended Data Fig. 6/extended_data_fig. 6fg.ipynb"
# - "Supplementary Fig. 6/supplementary_fig. 6.ipynb"
# - "Supplementary Fig. 7-9/supplementary_fig. 7-9.ipynb"
# - "Supplementary Fig. 13/supplementary_fig. 13.ipynb"
```

### Python Scripts
```bash
cd "Extended Data Fig. 6" && python extended_data_fig. 6de.py
```

### R Scripts
```bash
cd "Extended Data Fig. 5" && Rscript extended_data_fig. 5ab.R
```

## Output Files

Each script generates publication-quality figure files (PDF and/or PNG):
- Main figures: `fig.2a.*.pdf`, `fig.2b.*.pdf`, etc.
- Extended data figures: `extended_data_fig.*.pdf`
- Supplementary figures: `supplementary_fig.*.pdf`

## Model Comparison

The manuscript compares Phytomni against state-of-the-art AI models including general-purpose LLMs and specialized systems. Specific model comparisons are documented within each figure's notebook.

## Help

Please post in the GitHub issues or contact the authors with any questions about the repository, requests for more data, or additional information about the results.
