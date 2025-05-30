# Code for baselines of the TREC iKAT track

**[iKAT 2025]** Please use this other repository for more complete baselines, using SPLADE [https://github.com/SimonLupart/ikat-baseline](https://github.com/SimonLupart/ikat-baseline).

The code below gives a baseline for generating a run file for the TREC iKAT track. Using BM25 retrieval and MSMARCO miniLM rerank, with gpt4o for rewrite and answer generation.

## Recreating the Conda Environment

To recreate the conda environment used for this project, follow these steps:

1. Create the conda environment from the `environment.yml` file:
   ```bash
    conda env create -f env.yml
   ```
2. Activate the new environment:
   ```bash
    conda activate ikat24
   ```

## Usage

First, generating rewritten queries
```bash
python rewrite_gpt.py
```
then retrieval and reranking
```bash
python run_gpt4o_ikat24.py
```
then generating responses based on the ranking
```bash
python answer_gpt.py
```
convert to the format following the guidelines
```bash
python convert.py
```

Note: This baseline can also be run on the TREC iKAT 2023, 2024 or 2025 topics, and evaluated using the qrel file from respective year.

## Citations

Please cite our TREC iKAT overview paper if you use this work:

```bibtex
@inproceedings{coordinators-trec2024-papers-proc-4,
    author = {Mohammad Aliannejadi (University of Amsterdam), Zahra Abbasiantaeb (University of Amsterdam), Simon Lupart (University of Amsterdam), Shubham Chatterjee (University of Edinburgh), Jeffrey Dalton (University of Edinburgh), Leif Azzopardi (University of Strathclyde)},
    title = {TREC iKAT 2024: The Interactive Knowledge Assistance Track Overview},
    booktitle = {The Thirty-Third Text REtrieval Conference Proceedings (TREC 2024), Gaithersburg, MD, USA, November 15-18, 2024},
    series = {NIST Special Publication},
    volume = {1329},
    publisher = {National Institute of Standards and Technology (NIST)},
    year = {2024},
    trec_org = {coordinators},
    trec_runs = {},
    trec_tracks = {ikat}
   url = {https://trec.nist.gov/pubs/trec33/papers/Overview_ikat.pdf}
}
```


