# Code for baselines of the TREC iKAT 2024 track

This code gives a baseline for generating a run file for TREC iKAT 2024 track.

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

Note: This baseline can also be runned on the TREC iKAT 2023 topics, and evaluated using the qrel file from last year.

## License
This project is licensed under the MIT License.


