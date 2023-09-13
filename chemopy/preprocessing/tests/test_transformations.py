import unittest

import numpy as np
from sklearn.utils.estimator_checks import (
    check_estimators_unfitted,
    check_transformer_general,
)

from chemopy.preprocessing import AbsoluteValues, TransmittanceToAbsorbance


class TransmittanceToAbsorbanceTest(unittest.TestCase):
    def setUp(self):
        self.transformer = TransmittanceToAbsorbance()
        self.X = np.array([[0.5, 0.1, 0.01]])

    def test_fit(self):
        # The transformer doesn't have any trainable parameters,
        # so fit method should return self without any change.
        self.assertEqual(self.transformer.fit(self.X), self.transformer)

    def test_transform(self):
        transformed_X = self.transformer.fit_transform(self.X)
        expected_output = np.array([[0.301, 1, 2]])  # Expected absorbance values
        np.testing.assert_array_almost_equal(transformed_X, expected_output, decimal=3)

    def test_transform_unfitted(self):
        # Check that the transformer raises an error if transform is called before fit
        self.transformer = TransmittanceToAbsorbance()
        check_estimators_unfitted("TransmittanceToAbsorbance", self.transformer)

    def test_transformer_conformance(self):
        self.transformer = TransmittanceToAbsorbance().fit(self.X)
        # Check the transformer against scikit-learn's general transformer tests
        check_transformer_general("TransmittanceToAbsorbance", self.transformer)


class AbsoluteValuesTest(unittest.TestCase):
    def setUp(self):
        self.transformer = AbsoluteValues()
        self.X = np.array([[-1, -2, 3], [-4, 5, -6]])

    def test_fit(self):
        # The transformer doesn't have any trainable parameters,
        # so fit method should return self without any change.
        self.assertEqual(self.transformer.fit(self.X), self.transformer)

    def test_transform(self):
        transformed_X = self.transformer.fit_transform(self.X)
        expected_output = np.array([[1, 2, 3], [4, 5, 6]])  # Expected absorbance values
        np.testing.assert_equal(transformed_X, expected_output)

    def test_transform_unfitted(self):
        # Check that the transformer raises an error if transform is called before fit
        self.transformer = AbsoluteValues()
        check_estimators_unfitted("AbsoluteValues", self.transformer)

    def test_transformer_conformance(self):
        self.transformer = TransmittanceToAbsorbance().fit(self.X)
        # Check the transformer against scikit-learn's general transformer tests
        check_transformer_general("AbsoluteValues", self.transformer)


if __name__ == "__main__":
    unittest.main()
