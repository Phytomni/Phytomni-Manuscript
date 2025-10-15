# Phytomni Manuscript Code

Phytomni: Reproducibility code and notebooks for "An agentic AI for scientific discovery and design in plant research"

## Overview

This repository provides the code, scripts, and notebooks to reproduce figures, tables, and quantitative results reported in the manuscript “Phytomni: An agentic AI for scientific discovery and design in plant research.” Phytomni is a domain-specific, LLM-powered multi-agent system built on the Model Context Protocol (MCP) that integrates a plant-focused full-text knowledge base (~4.0M publications plus abstracts and patents), multi-omics data spanning 65 species, and 125 bioinformatics tools. The platform orchestrates hierarchically coordinated agents—Knowledge, Data, Analyst—and composite agents (e.g., In Silico Research Agent, Deep Genome Agent, Gene Network Agent, Digital Design Agent) to automate literature-grounded reasoning, data retrieval, and end-to-end bioinformatic analyses.

Code in this repository is organized by figure directory and has been tested with Python 3.8+ and R 4.0+. The shared dataset (data.xlsx) contains multi-sheet inputs for benchmarking and plotting (e.g., PhytoBench-Knowledge-ID/Trace, PhytoBench-Data, PhytoBench-Analysis, PhytoBench-Paper, and model-by-task performance). Executing the provided Python/R scripts and Jupyter notebooks reproduces the main and supplementary figures, including: (i) Knowledge Agent benchmarks (ID and Trace) versus state-of-the-art models (GPT-5, o3, Gemini-2.5-Pro, Claude-Opus-4.1, Grok-3-Beta, DeepSeek-V3, DeepSeek-R1); (ii) Data Agent natural-language-to-SQL performance on plant multi-omics; (iii) Analyst Agent goal-completion across diverse bioinformatics workflows; (iv) In Silico Research Agent paper-replication efficiency relative to a human expert baseline; and (v) Deep Genome Agent functional summarization and confabulation analyses. The outputs include publication-trend summaries and comparative heatmaps used throughout the manuscript.

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

## Data

All data is stored in `data.xlsx` with multiple sheets corresponding to different analyses:
- `GeneTuring` - Gene Turing test results
- `pangu` - Pangu model performance data  
- `other` - Comparative model performance data
- Model-specific sheets for individual analyses

## Code for Results/Figures

The directories correspond to the following figures/analyses:
- `Fig. 2` - Main figures comparing model performance (Jupyter notebook with Plotly)
- `Fig. S16` - Supplementary figures showing publication trends and document types (R scripts)
- `Fig. S22` - Supplementary figures with accuracy heatmaps and comparative model performance (Python script)
- `Fig. S23` - Additional supplementary analyses (Jupyter notebook)

### Running the Code

#### Python Scripts
```bash
cd "Fig. S22" && python plot.py
```

#### R Scripts  
```bash
cd "Fig. S16" && Rscript plot_figure_s16AB.R
```

#### Jupyter Notebooks
```bash
jupyter lab
# Then open and run cells in notebooks like "Fig. 2/fig. 2.ipynb"
```

## Output Files

Each script generates PDF and PNG figure files:
- Main figures: `fig.2a.*.pdf`, `fig.2b.*.pdf`, etc.
- Supplementary figures: `GeneTuring.pdf`, `pangu.pdf`, `other_models.pdf`
- Publication trends: `figure_s16_A.pdf`, `figure_s16_B.pdf`

## Model Comparison

The manuscript compares the following AI models:
- **Phytomni models**: Phyto-Reasoner, Phyto-Chatbot
- **General AI models**: GPT-4.1, Claude-3.7-Sonnet, Gemini-2.5-Pro, Grok-3-Beta
- **Open-source models**: Deepseek-V3, Deepseek-R1, o3

## Help

Please post in the Github issues or contact the authors with any questions about the repository, requests for more data, or additional information about the results.