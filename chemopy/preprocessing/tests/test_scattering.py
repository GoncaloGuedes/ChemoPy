import unittest

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import check_transformer_general

from chemopy.preprocessing import SNV


class SNVTests(unittest.TestCase):
    def test_fit_returns_self(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        snv = SNV()
        fitted_snv = snv.fit(X)
        self.assertIs(fitted_snv, snv)

    def test_transform(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        expected_result = np.array(
            [[-1.22474487, 0.0, 1.22474487], [-1.22474487, 0.0, 1.22474487]]
        )
        snv = SNV()
        transformed_X = snv.fit_transform(X)
        np.testing.assert_array_almost_equal(transformed_X, expected_result)

    def test_pipeline(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        expected_result = np.array(
            [[-1.22474487, 0.0, 1.22474487], [-1.22474487, 0.0, 1.22474487]]
        )
        pipeline = make_pipeline(SNV())
        transformed_X = pipeline.fit_transform(X)
        np.testing.assert_array_almost_equal(transformed_X, expected_result)

    def test_transformer_geral(self):
        snv = SNV()
        check_transformer_general("SNV", snv)


if __name__ == "__main__":
    unittest.main()
