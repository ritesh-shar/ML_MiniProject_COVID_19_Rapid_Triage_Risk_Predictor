# COVID-19 Outcome Prediction Using Radiomics and Clinical Data

A machine learning mini-project based on the research work *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients* by Xianghao Zhan and Yiheng Li.

By : 

Prathik M Nmabiar (PES2UG24CS371)

Ritesh Kumar Sharma (PES2UG24CS404)


## 1. Problem Statement

COVID-19 patients can experience different clinical outcomes, including admission to an intensive care unit (ICU), requiring mechanical ventilation (MV), or in-hospital death. Identifying patients at higher risk of these outcomes may help clinicians prioritize care and allocate medical resources.

This project explores a pretrained machine learning pipeline that uses CT-derived radiomics features and, when available, clinical and laboratory data to predict three outcomes:

- ICU admission
- Requirement for mechanical ventilation
- In-hospital death

The project also improves the reliability of the inference pipeline through input validation, missing-value imputation, patient-level feature aggregation, and prediction probability reporting.

**Important:** This is an academic machine learning project, not a clinically validated diagnostic or decision-support tool.

## 2. Dataset Details

### Available data

The repository currently includes example input files and pretrained models:

| File | Purpose |
|---|---|
| `example_MLinput.csv` | Example CT-derived radiomics features |
| `example_lab_input.csv` | Example clinical and laboratory features |
| `example_median.csv` | Reference values used for missing-value imputation |
| `models/` | Pretrained classifiers for the three prediction outcomes |

The supplied example radiomics and clinical/laboratory files represent one example patient. They are intended to demonstrate the inference pipeline, not to serve as a training or evaluation dataset.

### Original research dataset

The original study reports model development using a cohort of 1,662 patients and evaluation on a separate cohort of 700 patients. The paper reports AUROC values of 0.911 for ICU admission, 0.921 for mechanical ventilation, and 0.834 for death for its selected combined-data models.

These figures are results reported by the original researchers. They are not measurements obtained by independently evaluating this project's modified code.

The original patient-level dataset and outcome labels are not included in the available project files. Therefore, independent evaluation and reproduction of the original results are outside the scope of the current implementation.

## 3. Approach

The project uses pretrained classifiers to predict each outcome independently.

### A. Radiomics-only prediction

The pipeline processes CT-derived radiomics features and predicts ICU admission, mechanical ventilation, and death.

### B. Combined radiomics and clinical/laboratory prediction

When clinical and laboratory data are supplied, the pipeline combines the radiomics features with the corresponding patient information and uses the pretrained combined-data classifiers.

### C. Input validation

Validation checks help reject invalid inputs before prediction. These include empty input files, missing or blank patient IDs, duplicate clinical/laboratory patient IDs, mismatched patient IDs across inputs, and invalid median reference data.

### D. Missing-value imputation

Missing or nonnumeric clinical/laboratory feature values are converted to missing values and imputed using the corresponding reference median, provided that a valid median is available.

### E. Patient-level radiomics aggregation

When multiple lesion rows belong to the same patient, lesion-level features are aggregated into patient-level statistics. The aggregation includes mean, standard deviation, median, skewness, and the 25th and 75th percentiles. Invalid numeric values are handled to produce a consistent patient-level feature table.

### F. Prediction probability reporting

The inference pipeline reports both the predicted class and the model's estimated probability for class 1. The probability column is selected using the classifier's actual class labels rather than assuming a fixed column order.

These probabilities are model outputs and should not be interpreted as calibrated clinical risks without further evaluation.

## 4. Implementation

The project is implemented in Python using pandas, NumPy, scikit-learn, LightGBM, and supporting image-processing libraries.

Key files include:

- `COVID-19_prediction.py` — main inference pipeline, validation, imputation, aggregation, and probability reporting.
- `proc_radiomic_feature.py` — radiomics feature extraction workflow.
- `radiom_utils.py` — radiomics utility functions.
- `models/` — pretrained prediction models.
- `test_input_validation.py` — radiomics input-validation tests.
- `test_lab_validation.py` — clinical/laboratory validation and imputation tests.
- `test_patient_aggregation.py` — patient-level aggregation tests.
- `test_prediction_analysis.py` — prediction probability-analysis tests.

### Testing

The current implementation passes **26 unit tests** covering:

- Radiomics input validation
- Clinical/laboratory input validation
- Missing-value imputation
- Patient-level lesion aggregation
- Prediction probability handling

These tests verify the software behaviors covered by the test cases. They do not establish clinical predictive accuracy.

## 5. Environment Setup

The project has been tested in a Windows environment using Conda, Python 3.6.13, and the dependency versions specified for this project.

### Step 1: Clone the repository

```bash
git clone https://github.com/ritesh-shar/ML_MiniProject_COVID_19_Rapid_Triage_Risk_Predictor.git
cd ML_MiniProject_COVID_19_Rapid_Triage_Risk_Predictor
```

If you already have the repository cloned, navigate to its local directory instead.

### Step 2: Create and activate the environment

```bash
conda create -n covid-ml python=3.6.13
conda activate covid-ml
```

### Step 3: Install dependencies

Install the project's pinned dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The original pipeline also depends on compatible versions of the radiomics, imaging, and machine learning libraries. If installation fails, use the tested environment and dependency versions documented by the project rather than upgrading libraries indiscriminately.

## 6. Running the Project

Run commands from the repository's root directory with the `covid-ml` environment activated.

### Radiomics-only prediction

```bash
python COVID-19_prediction.py --radiomics_data example_MLinput.csv
```

### Combined radiomics and clinical/laboratory prediction

```bash
python COVID-19_prediction.py --radiomics_data example_MLinput.csv --lab_data example_lab_input.csv
```

The combined pipeline uses the reference median data expected by the implementation for clinical/laboratory imputation.

### Run the unit tests

```bash
python -m unittest discover -v
```

The current test suite contains 26 passing tests.

## 7. Results and Interpretation

The pipeline produces three predictions for each supported mode:

- ICU admission
- Mechanical ventilation
- Death

For each outcome, it reports the predicted class and estimated class-1 probability.

A predicted class of `0` indicates that the classifier selected class 0. The clinical interpretation depends on the original model's label encoding and training setup. It does not guarantee that an outcome will or will not occur.

The example files are used to demonstrate execution. Their outputs must not be treated as evidence of clinical accuracy or as medical advice.

## 8. Limitations and Future Work

The current project has several limitations:

1. **Limited data availability:** The available example CSV files do not provide a labeled patient cohort suitable for independent model evaluation.
2. **No independent performance evaluation:** Metrics such as AUROC, sensitivity, specificity, precision, and calibration cannot be independently calculated without suitable labeled test data.
3. **Pretrained models:** The project uses existing model artifacts rather than retraining and validating the original models.
4. **Probability calibration:** The reported probabilities have not been independently validated as calibrated clinical risk estimates.
5. **Generalizability:** The original study used early-pandemic data from a specific patient population. Its results should not automatically be generalized to other populations or current clinical settings.

Future work could include obtaining authorized access to labeled data, performing an independent evaluation, examining calibration, and assessing generalizability across patient populations.

## 9. Conclusion

This project demonstrates an end-to-end inference workflow for COVID-19 outcome prediction using CT-derived radiomics features and optional clinical/laboratory data. Engineering improvements address input validation, missing-value handling, lesion-to-patient aggregation, and probability reporting. Unit tests verify the implemented software behaviors, while the absence of the original labeled patient dataset limits independent evaluation of predictive performance.

The project provides a foundation for studying machine learning inference pipelines, robust data preprocessing, testing, and responsible interpretation of model outputs.

## 10. References

Zhan, X., and Li, Y. *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients.* CS229 Final Report, Stanford University.

Original repository cited by the paper: https://github.com/terryli710/COVID-19_prediction

Repository used for this mini-project: https://github.com/ritesh-shar/ML_MiniProject_COVID_19_Rapid_Triage_Risk_Predictor
