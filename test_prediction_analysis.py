
import unittest
import numpy as np

from importlib.machinery import SourceFileLoader

prediction_module = SourceFileLoader(
    "prediction_module",
    "COVID-19_prediction.py"
).load_module()

predict_with_probability = (
    prediction_module.predict_with_probability
)


class FakeClassifier:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([1, 0])

    def predict_proba(self, features):
        return np.array([
            [0.18, 0.82],
            [0.75, 0.25]
        ])


class FakeClassifierReversedClasses:
    classes_ = np.array([1, 0])

    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        # Column order follows classes_: [1, 0].
        return np.array([[0.82, 0.18]])


class NoProbabilityClassifier:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([1])


class PredictionAnalysisTests(unittest.TestCase):

    def test_returns_predictions_and_class_one_probabilities(self):
        model = FakeClassifier()

        predictions, probabilities = predict_with_probability(
            model, np.zeros((2, 1))
        )

        np.testing.assert_array_equal(predictions, [1, 0])
        np.testing.assert_allclose(probabilities, [0.82, 0.25])

    def test_uses_class_labels_not_assumed_column_order(self):
        model = FakeClassifierReversedClasses()

        predictions, probabilities = predict_with_probability(
            model, np.zeros((1, 1))
        )

        np.testing.assert_array_equal(predictions, [1])
        np.testing.assert_allclose(probabilities, [0.82])

    def test_rejects_models_without_probability_support(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            predict_with_probability(
                NoProbabilityClassifier(), np.zeros((1, 1))
            )

    def test_rejects_unknown_positive_class(self):
        with self.assertRaisesRegex(ValueError, "not available"):
            predict_with_probability(
                FakeClassifier(),
                np.zeros((1, 1)),
                positive_class=2
            )


if __name__ == "__main__":
    unittest.main()
