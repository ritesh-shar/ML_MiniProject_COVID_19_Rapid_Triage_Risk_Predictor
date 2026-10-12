import unittest
import pandas as pd
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("covid_prediction", "COVID-19_prediction.py")
covid_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covid_prediction)

lesion_to_patient = covid_prediction.lesion_to_patient


class TestPatientAggregation(unittest.TestCase):

    def test_multiple_lesions_aggregate_to_one_patient_row(self):
        data = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2"],
            "feature_a": [1.0, 3.0, 10.0],
            "log-sigma-2-0-mm-3D_firstorder_Skewness": [0.2, 0.4, 0.6]
        })
        result = lesion_to_patient(data)
        self.assertEqual(result.shape[0], 2)
        self.assertAlmostEqual(result.loc["p1", "feature_a_mean"], 2.0)
        self.assertAlmostEqual(result.loc["p1", "feature_a_medium"], 2.0)
        self.assertEqual(result.loc["p1", "lesion count"], 2)
        self.assertEqual(result.loc["p2", "lesion count"], 1)

    def test_single_lesion_statistics_are_filled(self):
        data = pd.DataFrame({
            "patient_id": ["p1"],
            "feature_a": [7.0],
            "log-sigma-2-0-mm-3D_firstorder_Skewness": [0.5]
        })
        result = lesion_to_patient(data)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.loc["p1", "feature_a_mean"], 7.0)
        self.assertEqual(result.loc["p1", "feature_a_std"], 0.0)
        self.assertEqual(result.loc["p1", "feature_a_skew"], 0.0)
        self.assertEqual(result.loc["p1", "lesion count"], 1)

    def test_nonnumeric_feature_is_coerced_and_missing_statistics_filled(self):
        data = pd.DataFrame({
            "patient_id": ["p1", "p1"],
            "feature_a": ["2.5", "not-a-number"]
        })
        result = lesion_to_patient(data)
        self.assertAlmostEqual(result.loc["p1", "feature_a_mean"], 2.5)
        self.assertEqual(result.loc["p1", "feature_a_std"], 0.0)
        self.assertEqual(result.loc["p1", "lesion count"], 2)

    def test_missing_patient_id_rejected(self):
        data = pd.DataFrame({"patient_id": ["p1", None], "feature_a": [1, 2]})
        with self.assertRaisesRegex(ValueError, "missing patient_id"):
            lesion_to_patient(data)

    def test_missing_patient_id_column_rejected(self):
        data = pd.DataFrame({"feature_a": [1, 2]})
        with self.assertRaisesRegex(ValueError, "patient_id"):
            lesion_to_patient(data)

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            lesion_to_patient(pd.DataFrame())

    def test_input_is_not_mutated(self):
        data = pd.DataFrame({
            "patient_id": ["p1"],
            "feature_a": [3.0]
        })
        original_columns = list(data.columns)
        lesion_to_patient(data)
        self.assertEqual(list(data.columns), original_columns)
        self.assertEqual(len(data.columns), 2)


if __name__ == "__main__":
    unittest.main()
