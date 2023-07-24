import unittest
import numpy as np
from sklearn.utils.estimator_checks import check_transformer_general, check_estimators_unfitted
from chemopy.preprocessing import ScaleMaxMin, Centering


class ScaleMaxMinTest(unittest.TestCase):
    def setUp(self):
        self.transformer = ScaleMaxMin()
        self.X = np.array([[1, 2, 3],
                           [4, 5, 6]
                           ])
    def test_transform_unfitted(self):
        # Check that the transformer raises an error if transform is called before fit
        self.transformer = ScaleMaxMin()
        check_estimators_unfitted("ScaleMaxMin", self.transformer)
        
    def test_fit(self):
        # The transformer doesn't have any trainable parameters,
        # so fit method should return self without any change.
        self.assertEqual(self.transformer.fit(self.X), self.transformer)

    def test_transform(self):
        transformed_X = self.transformer.fit_transform(self.X)
        expected_output = np.array([[0, 0.5, 1],
                                    [0, 0.5, 1]
                                    ])  # Expected absorbance values
        np.testing.assert_array_almost_equal(transformed_X, expected_output, decimal=3)

    def test_transformer_conformance(self):
        self.transformer = ScaleMaxMin().fit(self.X)
        # Check the transformer against scikit-learn's general transformer tests
        check_transformer_general("ScaleMaxMin", self.transformer)
        
    def test_min_value_greater(self):
        with self.assertRaises(ValueError):
            ScaleMaxMin(1, 0)


class CenteringTest(unittest.TestCase):
    def setUp(self):
        self.transformer = Centering()
        self.X = np.array([[1, 2, 3],
                           [4, 5, 6]
                           ])
    def test_transform_unfitted(self):
        # Check that the transformer raises an error if transform is called before fit
        self.transformer = Centering()
        check_estimators_unfitted("CenteringTest", self.transformer)
        
    def test_fit(self):
        # The transformer doesn't have any trainable parameters,
        # so fit method should return self without any change.
        self.assertEqual(self.transformer.fit(self.X), self.transformer)

    def test_transform_mean(self):
        transformed_X = self.transformer.fit_transform(self.X)
        expected_output = np.array([[-1.5, -1.5, -1.5],
                                    [1.5, 1.5, 1.5]
                                    ])  # Expected absorbance values
        np.testing.assert_array_almost_equal(transformed_X, expected_output, decimal=2)
    
    def test_transform_median(self):
        transformer = Centering("median")
        transformed_X = transformer.fit_transform(self.X)
        expected_output = np.array([[-1.5, -1.5, -1.5],
                                    [1.5, 1.5, 1.5]
                                    ])  # Expected absorbance values
        np.testing.assert_array_almost_equal(transformed_X, expected_output, decimal=2)

    def test_transformer_conformance(self):
        self.transformer = Centering()
        # Check the transformer against scikit-learn's general transformer tests
        check_transformer_general("CenteringTest", self.transformer)

    def test_incorrect_method(self):
        with self.assertRaises(ValueError):
            transformer = Centering("incorrect_method")
    

if __name__ == "__main__":
    unittest.main()
