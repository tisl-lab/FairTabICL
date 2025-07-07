# Towards Fair In-Context Learning with Tabular Foundation Models

This repository contains code for running fairness experiments with different models and selection methods, focusing on tabular in-context learning approaches. Read the paper: [https://arxiv.org/pdf/2505.09503](https://arxiv.org/pdf/2505.09503)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Running Experiments

You can run experiments for single evaluation or fairness-accuracy :

- Use `src/run.py` for single fairness evaluation with different preprocessings  (details below)
- Use `src/Tradeoff_Evaluation.py` for fairness-accuracy tradeoff evaluation.

## Data Preprocessing

The `preprocessing` folder contains preprocessed datasets used in the experiments. The ACS PUMS datasets will be automatically downloaded from the Folktable benchmark when needed.

## Command-line experiment

The main script `src/run.py` allows you to run single fairness experiments with various configurations. Here's how to use it:

### Basic Usage

```bash
python src/run.py --dataset <dataset_name> --base_model <model> --selection_method <method>
```

### Available Options


#### `dataset`: Datasets 
- `adult_income`: ACSIncome - Predicts if income exceeds $50,000 for individuals over 16 who work at least 1 hour/week
- `employment`: ACSEmployment - Predicts employment status for individuals aged 16-90
- `mobility`: ACSMobility - Predicts if an individual had the same address one year ago (filtered to ages 18-35)
- `coverage`: ACSPublicCoverage - Predicts if an individual has public health insurance (filtered to under 65 with income below $30,000)
- `travel`: ACSTravelTime - Predicts if commute time exceeds 20 minutes for employed individuals over 16
- `celeba_attract`: CelebA - Predicts attractiveness using facial attributes with gender as sensitive attribute
- `diabetes_race`: Diabetes - Predicts diabetes diagnosis using health, lifestyle and demographic features
- `german_age`: German Credit - Predicts credit risk using age (over/under 25) as sensitive attribute

#### `base_model`: Fondation Models
- `tabpfn`: [TabPFN](https://www.nature.com/articles/s41586-024-08328-6). Available at [TabPFN GitHub Repository](https://github.com/PriorLabs/TabPFN)
- `tabicl`: [TabICL](https://arxiv.org/abs/2502.05564). Available at [TabICL GitHub Repository](https://github.com/soda-inria/tabicl)

#### `selection_method`: Preprocessing fairness intervention method
- `vanilla`: Selects examples randomly without fairness intervention.
- `uncertain`: Selects examples with uncertain predictions. 
- `group_balanced`: Ensures equal representation across groups.
- `class_group_balanced`: Ensures equal representation across groups x classes.
- `correlation_remover`: Apply [Correlation Remover](https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.CorrelationRemover.html). The argument `cr_sanitization_strategy` Controls how correlation removal is applied (default: 1).
    - `1`: Apply transformation to both training and testing data, this variant tends to amplify unfairness. 
    - `2`: Apply transformation only to the testing data, this variant yields better fairness performance.



### Additional Parameters

- `--seed`: Random seed for reproducibility (default: 1)
- `--cp_alpha`: parameter controlling the fairness accuracy tradeoff (default: 0.05). 
   - When `selection_method` is `uncertain` it controls the coverage level for conformal prediction (epsilon).
   - When `selection_method` is `correlation_remover` it controls the strength of correlation remover (alpha). 
- `--train_size`: Number of training examples to use (default: 1000), set -1 to use all training data available as in-context examples.
- `--exp_name`: Custom name for the experiment (for logging and saving results)
- `--base_model_uncertainty`: Model for uncertainty estimation (choices: "lr", "tabpfn")

### Example Commands

1. Basic experiment with vanilla method without fairness intervention TabPFN:
```bash
python src/run.py --dataset german_age --base_model tabpfn --selection_method vanilla
```

2. Experiment with uncertainty-based selection:
```bash
python src/run.py --dataset adult_income --base_model tabpfn --selection_method uncertain --cp_alpha 0.1
```

3. Experiment with group balanced sampling:
```bash
python src/run.py --dataset diabetes_race --base_model tabicl --selection_method group_balanced --train_size -1
```

4. Fairness-accuracy tradeoff evaluation: 

- Uncertainty-based method
```bash
python src/Tradeoff_Evaluation.py --dataset adult_income --base_model tabpfn --selection_method uncertain --base_model_uncertainty tabpfn
```
- Correlation Remover
```bash
python src/Tradeoff_Evaluation.py --dataset adult_income --base_model tabpfn --selection_method correlation_remover
```

## Results

Results are saved in the `analysis/results` directory with the following structure:
```
results/
└── <dataset>/
    └── t_size_<train_size>/
        └── <base_model>/
            └── <selection_method>/
                └── seed_<seed>.csv
```


Each CSV file contains the following metrics:

- Accuracy (acc)
- F1 Score (f1)
- Area Under Curve (auc)
- Demographic Parity (dp)
- Equal Opportunity (eop)
- Equalized Odds (eodds)

Results of the tradeoff evaluation are saved in the `analysis/tradeoff` directory with the same structure.

## Visualisation and plots of the paper: 

Once the code has been executed, use the notebook [visualization.ipynb](visualization.ipynb) to aggregate and reproduce the results of the paper. 


## Notes

- TabPFN has a maximum sample size limit of 10,000. If your training set exceeds this, it will be automatically subsampled.
- The results are saved in CSV format for easy analysis.
- Each experiment runs with 5-fold cross-validation by default.

## Citation

If you use (parts of) this code, please cite:

```
@article{kenfack2025towards,
  title={Towards Fair In-Context Learning with Tabular Foundation Models},
  author={Kenfack, Patrik and Kahou, Samira Ebrahimi and A{\"\i}vodji, Ulrich},
  journal={arXiv preprint arXiv:2505.09503},
  year={2025}
}```
