import unittest
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("covid_prediction", "COVID-19_prediction.py")
covid_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covid_prediction)

validate_lab_data = covid_prediction.validate_lab_data
impute_lab_data = covid_prediction.impute_lab_data


class TestLabValidationAndImputation(unittest.TestCase):

    def setUp(self):
        self.lab = pd.DataFrame({
            "patient_id": ["p1", "p2"],
            "age": [50, None],
            "wbc": [None, 6.2]
        })
        self.medians = pd.DataFrame({
            "patient_id": ["median"],
            "age": [55],
            "wbc": [5.4]
        })
        self.radiomics = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2"],
            "feature_a": [1, 2, 3]
        })

    def test_empty_lab_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_lab_data(pd.DataFrame(), median_data=self.medians)

    def test_missing_patient_id_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "patient_id"):
            validate_lab_data(pd.DataFrame({"age": [50]}))

    def test_missing_patient_id_value_rejected(self):
        bad = pd.DataFrame({"patient_id": ["p1", None], "age": [50, 60]})
        with self.assertRaisesRegex(ValueError, "missing patient_id"):
            validate_lab_data(bad)

    def test_blank_patient_id_rejected(self):
        bad = pd.DataFrame({"patient_id": ["   "], "age": [50]})
        with self.assertRaisesRegex(ValueError, "missing patient_id"):
            validate_lab_data(bad)

    def test_duplicate_patient_ids_rejected(self):
        bad = pd.DataFrame({"patient_id": ["p1", "p1"], "age": [50, 60]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_lab_data(bad)

    def test_patient_id_mismatch_rejected(self):
        bad_lab = pd.DataFrame({"patient_id": ["p1", "p3"], "age": [50, 60]})
        with self.assertRaisesRegex(ValueError, "Patient ID mismatch"):
            validate_lab_data(bad_lab, radiomics_data=self.radiomics)

    def test_missing_median_column_rejected(self):
        bad_medians = pd.DataFrame({"patient_id": ["median"], "age": [55]})
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_lab_data(self.lab, median_data=bad_medians)

    def test_missing_values_imputed_for_all_rows(self):
        actual = impute_lab_data(self.lab, self.medians)
        self.assertEqual(actual.loc[0, "age"], 50)
        self.assertEqual(actual.loc[1, "age"], 55)
        self.assertEqual(actual.loc[0, "wbc"], 5.4)
        self.assertEqual(actual.loc[1, "wbc"], 6.2)
        # Input is not modified in place.
        self.assertTrue(pd.isnull(self.lab.loc[1, "age"]))

    def test_nonnumeric_value_is_imputed(self):
        lab = pd.DataFrame({"patient_id": ["p1"], "age": ["unknown"]})
        medians = pd.DataFrame({"patient_id": ["median"], "age": [55]})
        result = impute_lab_data(lab, medians)
        self.assertEqual(result.loc[0, "age"], 55)

    def test_missing_median_for_missing_feature_rejected(self):
        lab = pd.DataFrame({"patient_id": ["p1"], "age": [None]})
        medians = pd.DataFrame({"patient_id": ["median"], "age": [None]})
        with self.assertRaisesRegex(ValueError, "Cannot impute"):
            impute_lab_data(lab, medians)


if __name__ == "__main__":
    unittest.main()
