# Convert input
import warnings
import pandas as pd
import numpy as np
import argparse
import joblib
warnings.simplefilter('ignore', UserWarning)


def validate_radiomics_data(data):
    """Validate radiomics input before patient-level aggregation."""
    if data.empty:
        raise ValueError("Radiomics input CSV is empty.")

    if "patient_id" not in data.columns:
        raise ValueError(
            "Radiomics input must contain a 'patient_id' column."
        )

    # Reject missing, empty, or whitespace-only patient IDs.
    patient_ids = data["patient_id"]

    if patient_ids.isnull().any() or patient_ids.astype(str).str.strip().eq("").any():
        raise ValueError(
            "Radiomics input contains missing patient_id values."
        )
def predict_with_probability(model, features, positive_class=1):
    """Return predicted classes and probabilities for a chosen class."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model does not support probability predictions."
        )

    if not hasattr(model, "classes_"):
        raise ValueError("The model does not expose its class labels.")

    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            "Requested class {} is not available in the model.".format(
                positive_class
            )
        )

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    if probabilities.ndim != 2:
        raise ValueError("Model probabilities must be a 2D array.")

    if probabilities.shape[1] != len(classes):
        raise ValueError(
            "Probability columns do not match the model's classes."
        )

    positive_index = classes.index(positive_class)

    return predictions, probabilities[:, positive_index]



# load file

def validate_lab_data(data, radiomics_data=None, median_data=None):
    """Validate patient-level clinical/laboratory data before imputation."""
    if data is None or data.empty:
        raise ValueError("Clinical/lab input CSV is empty.")

    if "patient_id" not in data.columns:
        raise ValueError("Clinical/lab input must contain a 'patient_id' column.")

    patient_ids = data["patient_id"]
    if patient_ids.isnull().any() or patient_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Clinical/lab input contains missing patient_id values.")

    if data["patient_id"].duplicated().any():
        duplicates = data.loc[data["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError("Clinical/lab input contains duplicate patient_id values: {}".format(duplicates))

    if radiomics_data is not None:
        if "patient_id" not in radiomics_data.columns:
            raise ValueError("Radiomics input must contain a 'patient_id' column.")
        radiomics_ids = set(radiomics_data["patient_id"].dropna().astype(str).str.strip())
        lab_ids = set(patient_ids.astype(str).str.strip())
        missing_lab = sorted(radiomics_ids - lab_ids)
        extra_lab = sorted(lab_ids - radiomics_ids)
        if missing_lab or extra_lab:
            raise ValueError(
                "Patient ID mismatch between radiomics and clinical/lab data. "
                "Missing lab IDs: {}; extra lab IDs: {}.".format(missing_lab, extra_lab)
            )

    if median_data is not None:
        if median_data.empty:
            raise ValueError("Median/imputation input is empty.")
        if "patient_id" not in median_data.columns:
            raise ValueError("Median/imputation input must contain a 'patient_id' column.")
        if len(median_data) != 1:
            raise ValueError("Median/imputation input must contain exactly one reference row.")
        missing_medians = sorted(set(data.columns) - set(median_data.columns) - {"patient_id"})
        if missing_medians:
            raise ValueError(
                "Median/imputation input is missing columns: {}.".format(missing_medians)
            )


def impute_lab_data(lab_data, lab_median):
    """Return a copy of lab data with missing feature values filled from median row."""
    if lab_data is None or lab_data.empty:
        raise ValueError("Clinical/lab input CSV is empty.")
    if lab_median is None or lab_median.empty:
        raise ValueError("Median/imputation input is empty.")
    if "patient_id" not in lab_data.columns or "patient_id" not in lab_median.columns:
        raise ValueError("Both clinical/lab and median data must contain 'patient_id'.")

    result = lab_data.copy()
    median_row = lab_median.iloc[0]
    feature_columns = [col for col in result.columns if col != "patient_id"]

    missing_medians = [col for col in feature_columns if col not in lab_median.columns]
    if missing_medians:
        raise ValueError("Median/imputation input is missing columns: {}.".format(missing_medians))

    for col in feature_columns:
        # Convert invalid numeric strings to NaN so they are handled consistently.
        result[col] = pd.to_numeric(result[col], errors="coerce")
        median_value = pd.to_numeric(pd.Series([median_row[col]]), errors="coerce").iloc[0]
        if result[col].isnull().any():
            if pd.isnull(median_value):
                raise ValueError(
                    "Cannot impute '{}' because its reference median is missing or nonnumeric.".format(col)
                )
            result[col] = result[col].fillna(median_value)
    return result


# YOU MAY CHANGE THIS LINE
radiomics_data = pd.read_csv("example_MLinput.csv")
# YOU MAY CHANGE THIS LINE
lab_data = pd.read_csv('example_lab_input.csv')

# YOU SHOULD NOT CHANGE THE FOLLOWING LINES
clinlab = pd.read_csv("example_median.csv")
del_col_name = ['ct_path','study_id','series_id','diagnostics_Versions_PyRadiomics','diagnostics_Versions_Numpy',
                'diagnostics_Versions_SimpleITK','diagnostics_Versions_PyWavelet','diagnostics_Versions_Python',
                'diagnostics_Configuration_Settings','diagnostics_Configuration_EnabledImageTypes',
                'diagnostics_Image-original_Hash','diagnostics_Image-original_Dimensionality',
                'diagnostics_Image-original_Spacing','diagnostics_Image-original_Size','diagnostics_Mask-original_Hash',
                'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_Size',
                'diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex',
                'diagnostics_Mask-original_CenterOfMass']
# All the numeric columns convert to numeric
def lesion_to_patient(data):
    """Aggregate lesion-level radiomics rows into one record per patient."""
    if data is None or data.empty:
        raise ValueError("Radiomics input is empty; cannot aggregate lesions.")
    if "patient_id" not in data.columns:
        raise ValueError("Radiomics input must contain a 'patient_id' column.")
    if data["patient_id"].isnull().any() or data["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Radiomics input contains missing patient_id values.")

    # Work on a copy and preserve feature order; do not mutate global column lists.
    data = data.copy()
    excluded = set(del_col_name + ["patient_id"])
    xcols = [col for col in data.columns if col not in excluded]
    if not xcols:
        raise ValueError("Radiomics input contains no usable feature columns.")

    for col in xcols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    grouped = data.groupby("patient_id", sort=True)[xcols]
    feature_name_list = ["mean", "std", "medium", "skew", "75", "25"]
    feature_list = [
        grouped.mean(),
        grouped.std(),
        grouped.median(),
        grouped.skew(),
        grouped.quantile(q=0.75),
        grouped.quantile(q=0.25)
    ]

    renamed_features = []
    for frame, statistic in zip(feature_list, feature_name_list):
        renamed_features.append(frame.rename(
            columns={name: name + "_" + statistic for name in xcols}
        ))

    result = pd.concat(renamed_features, axis=1)
    if "log-sigma-2-0-mm-3D_firstorder_Skewness" in data.columns:
        counts = data.groupby("patient_id")[
            "log-sigma-2-0-mm-3D_firstorder_Skewness"
        ].count()
    else:
        # Fall back to number of lesion rows if the legacy count feature is absent.
        counts = data.groupby("patient_id").size()
    counts.name = "lesion count"
    result = result.join(counts)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)

def radiomicsModel(radiomics_data):
    radiomics_data = pd.read_csv(radiomics_data)
    validate_radiomics_data(radiomics_data)
    patient_input = lesion_to_patient(radiomics_data)
    ##Radiomics Model
    mean_std = np.load("models/radiomics_ss.npy")
    feature_name = joblib.load("models/radiomics_feature_name.joblib")
    reordered = patient_input[feature_name]
    test_X_std = (reordered - mean_std[0,:]) / mean_std[1,:]

    ICU_clfmodel = joblib.load("models/radiomics_models_ICU_clfmodel.joblib")
    ICU_fsmodel = joblib.load("models/radiomics_models_ICU_fsmodel.joblib")
    ICU_x = ICU_fsmodel.transform(test_X_std)
    ICU, ICU_prob = predict_with_probability(
    ICU_clfmodel, ICU_x)

    MV_clfmodel = joblib.load("models/radiomics_models_MV_clfmodel.joblib")
    MV_x = test_X_std
    MV , MV_prob = predict_with_probability(
    MV_clfmodel, MV_x
)

    death_clfmodel = joblib.load("models/radiomics_models_death_clfmodel.joblib")
    death_fsmodel = joblib.load("models/radiomics_models_death_fsmodel.joblib")
    death_x = death_fsmodel.transform(test_X_std)
    death, death_prob = predict_with_probability(
    death_clfmodel, death_x
)

    print(
    'Radiomics Model:'
    '\nICU: {} (class 1 probability: {:.4f}%)'
    '\nMV: {} (class 1 probability: {:.6f}%)'
    '\nDeath: {} (class 1 probability: {:.4f}%)'.format(
        ICU[0], ICU_prob[0]*100,
        MV[0], MV_prob[0]*100,
        death[0], death_prob[0]*100
    )
)

def radiomicsClinLab(radiomics_data, lab_data):
    radiomics_data = pd.read_csv(radiomics_data)
    validate_radiomics_data(radiomics_data)
    lab_data = pd.read_csv(lab_data)
    patient_input = lesion_to_patient(radiomics_data)
    ##RadioClinLab Model
    mean_std = np.load("models/radioclinlab_ss.npy")
    feature_name = joblib.load("models/radiomics_feature_name.joblib")
    # Imputation
    lab_median = pd.read_csv('example_median.csv')
    lab_median = lab_median.drop(['patient_id'], axis=1)
    lab_data = lab_data.set_index('patient_id')
    columns = lab_median.columns
    for item in columns:
        if np.isnan(lab_data[item].values[0]):
            lab_data[item].values[0] = lab_median[item].values[0]

    reordered = patient_input[feature_name]
    patient_input_radioclinlab = pd.concat([reordered, lab_data], axis=1, ignore_index=True)
    test_X_std = (patient_input_radioclinlab - mean_std[0, :]) / mean_std[1, :]

    ICU_clfmodel = joblib.load("models/radioclinlab_models_ICU_clfmodel.joblib")
    ICU_fsmodel = joblib.load("models/radioclinlab_models_ICU_fsmodel.joblib")
    ICU_x = ICU_fsmodel.transform(test_X_std)
    ICU , ICU_prob = predict_with_probability(
    ICU_clfmodel, ICU_x
)
        

    MV_clfmodel = joblib.load("models/radioclinlab_models_MV_clfmodel.joblib")
    MV_fsmodel = joblib.load("models/radioclinlab_models_MV_fsmodel.joblib")
    MV_x = MV_fsmodel.transform(test_X_std)
    MV , MV_prob = predict_with_probability(
    MV_clfmodel, MV_x
)

    death_clfmodel = joblib.load("models/radioclinlab_models_death_clfmodel.joblib")
    death_fsmodel = joblib.load("models/radioclinlab_models_death_fsmodel.joblib")
    death_x = death_fsmodel.transform(test_X_std)
    death, death_prob = predict_with_probability(
    death_clfmodel, death_x
)

    print(
        'RadioClinLab Model:'
        '\nICU: {} (class 1 probability: {:.4f}%)'
        '\nMV: {} (class 1 probability: {:.6f}%)'
        '\nDeath: {} (class 1 probability: {:.4f}%)'.format(
            ICU[0], ICU_prob[0] * 100,
            MV[0], MV_prob[0] * 100,
            death[0], death_prob[0] * 100
        )
    )


def parse_opts():
    '''
    set the argument for the project
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--radiomics_data', default='', type=str, help='the path of radiomics data')
    parser.add_argument('--lab_data', default='', type=str, help='the path of lab data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # command line
    opt = parse_opts()
    radiomics_data = opt.radiomics_data
    lab_data = opt.lab_data
    if not lab_data:
        radiomicsModel(radiomics_data)
    else:
        radiomicsClinLab(radiomics_data, lab_data)
    # radiomicsModel('test/final_merge_feature.csv')
    # python COVID-19_prediction --radiomics_data <>
    # python COVID-19_prediction --radiomics_data <> --lab_data <>
