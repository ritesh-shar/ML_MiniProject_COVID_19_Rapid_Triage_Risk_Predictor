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
    del_col_name.append("patient_id")
    Xcols = list(set(data.columns) - set(del_col_name))
    data[Xcols] = data[Xcols].apply(pd.to_numeric)
    feature_name_list = ['mean', 'std', 'medium', 'skew', '75', '25']
    X_mean = data.groupby('patient_id')[Xcols].mean()
    X_std = data.groupby('patient_id')[Xcols].std()[Xcols]
    X_medium = data.groupby('patient_id')[Xcols].median()[Xcols]
    X_skew = data.groupby('patient_id')[Xcols].skew()[Xcols]
    X_75 = data.groupby('patient_id')[Xcols].quantile(q=0.75)[Xcols]
    X_25 = data.groupby('patient_id')[Xcols].quantile(q=0.25)[Xcols]
    feature_list = [X_mean, X_std, X_medium, X_skew, X_75, X_25]
    for i in range(len(feature_name_list)):
        name_dict = {}
        for name in Xcols:
            name_dict[name] = name + '_' + feature_name_list[i]
        feature_list[i] = feature_list[i].rename(columns=name_dict)
    X_final_no_count = pd.concat(feature_list, axis=1)
    lesion_count = data.groupby('patient_id')['log-sigma-2-0-mm-3D_firstorder_Skewness'].count()
    lesion_count.name = 'lesion count'
    X_final = X_final_no_count.merge(lesion_count, on='patient_id')
    return X_final.fillna(0)

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
            print('Impute: ', item)

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
