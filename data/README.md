# Dataset Setup

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

## Required files

Download the dataset and place the following files inside `data/raw/`:

```text
data/raw/
├── diabetic_data.csv
└── IDS_mapping.csv
```

The modelling notebook expects `diabetic_data.csv` as the main encounter-level dataset. `IDS_mapping.csv` contains descriptive mappings for selected coded fields.

## Source

- Dataset: Diabetes 130-US Hospitals for Years 1999–2008
- Repository: UCI Machine Learning Repository
- DOI: https://doi.org/10.24432/C5230J

The raw dataset files are not redistributed in this repository. Download them directly from UCI and retain the original filenames.

## Data-use note

The dataset contains de-identified historical hospital encounter records. This repository is for portfolio and educational use only and must not be used for clinical decision-making.
