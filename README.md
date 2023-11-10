# Loan Data Processing and Evaluation

This repository hosts Python scripts and Jupyter Notebooks dedicated to processing, training, and evaluating machine learning models on loan data.

## Project Structure

- `analysis_outputs`: Contains model evaluation outputs like confusion matrix images and model weights files.
- `data`: Holds the raw `loan_data.csv` and the produced `preprocessed_data.csv`.
- `modules`: A Python package with `preprocess.py` and `train_evaluate.py` scripts.
- `notebooks`: Includes Jupyter Notebooks such as `eda.ipynb` for exploratory data analysis.
- `environment.yaml`: A Conda environment file for environment setup.
- `main.py`: The main script to run the entire processing and evaluation pipeline.
- `README.md`: The guide documenting the repository setup and execution instructions.

## Running Modules

### Preprocessing

The preprocessing module can be run with or without specifying arguments. By default, it will look for `loan_data.csv` in the `data` directory and output `preprocessed_data.csv` in the same directory. 

**Run with default settings:**

```bash
python modules/preprocess.py
```

To specify input and output paths for the preprocessing module, use the following command:

```bash
python modules/preprocess.py --input_path data/loan_data.csv --output_dir data
```

### Training and Evaluation

The training and evaluation module can be executed without providing explicit arguments. It defaults to use `preprocessed_data.csv` from the `data` directory and saves outputs to `analysis_outputs`. To run with defaults, use the following command:

```bash
python modules/train_evaluate.py
```

To specify input and output paths for the training and evaluation module, use the following command:

```bash
python modules/train_evaluate.py --input_path data/preprocessed_data.csv --output_dir analysis_outputs
```

### Full Pipeline

To execute the complete pipeline, including preprocessing, training, and evaluation, use the `main.py` script. While arguments are optional thanks to default values, they can be provided for custom data locations. To run with defaults, use the following command:

```bash
python main.py
```

To specify all optional arguments for the full pipeline, use the following command:

```bash
python main.py --input_path data/loan_data.csv --preprocess_output_dir data --train_output_dir analysis_outputs
```

### Optional Arguments

Arguments are optional for individual modules. If not specified, default paths within the project structure are used:

- `--input_path`: Path to the input loan data file (default is `data/loan_data.csv`).
- `--preprocess_output_dir`: Directory for saving preprocessed data (default is `data`).
- `--train_output_dir`: Directory for saving training and evaluation outputs (default is `analysis_outputs`).

Ensure the environment specified in `environment.yaml` is active to have all necessary dependencies installed before running the scripts. To create and activate the environment, use the following commands:

```bash
conda env create -f environment.yaml
conda activate wealthfront
```
