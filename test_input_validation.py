
import unittest
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location(
    "covid_prediction", "COVID-19_prediction.py"
)
covid_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(covid_prediction)

validate_radiomics_data = covid_prediction.validate_radiomics_data


class TestRadiomicsValidation(unittest.TestCase):

    def test_empty_dataframe_is_rejected(self):
        data = pd.DataFrame(columns=["patient_id", "age"])

        with self.assertRaisesRegex(ValueError, "empty"):
            validate_radiomics_data(data)

    def test_missing_patient_id_column_is_rejected(self):
        data = pd.DataFrame({"age": [45]})

        with self.assertRaisesRegex(ValueError, "patient_id"):
            validate_radiomics_data(data)

    def test_missing_patient_id_value_is_rejected(self):
        data = pd.DataFrame({
            "patient_id": [None],
            "age": [45]
        })

        with self.assertRaisesRegex(ValueError, "missing patient_id"):
            validate_radiomics_data(data)

    def test_valid_input_is_accepted(self):
        data = pd.DataFrame({
            "patient_id": ["patient_001"],
            "age": [45]
        })

        # Should not raise an exception.
        validate_radiomics_data(data)

    def test_whitespace_patient_id_is_rejected(self):
        data = pd.DataFrame({
            "patient_id": ["   "],
            "age": [45]
        })

        with self.assertRaises(ValueError):
            validate_radiomics_data(data)


if __name__ == "__main__":
    unittest.main()
