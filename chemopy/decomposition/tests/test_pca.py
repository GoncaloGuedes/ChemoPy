import unittest

import numpy as np

from chemopy.decomposition import PCA


class TestPCA(unittest.TestCase):

    # PCA can be initialized with default parameters
    def test_initialized_with_default_parameters(self):
        pca = PCA()
        self.assertEqual(pca.n_components, 2)
        self.assertEqual(pca.mean_center, False)
        self.assertEqual(pca.confidence_level, 0.95)

    # PCA can be initialized with custom parameters
    def test_initialized_with_custom_parameters(self):
        pca = PCA(n_components=3, mean_center=True, confidence_level=0.99)
        self.assertEqual(pca.n_components, 3)
        self.assertEqual(pca.mean_center, True)
        self.assertEqual(pca.confidence_level, 0.99)

    # PCA throws an error when fitting an array with less than 5 samples
    def test_fit_less_than_5_samples(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            pca.fit(data)

    # PCA can fit a numpy array
    def test_fit_numpy_array(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        pca.fit(data)
        self.assertIsNotNone(pca.loadings_)
        self.assertIsNotNone(pca.explained_variance_)
        self.assertIsNotNone(pca.explained_variance_accumulative)
        self.assertIsNotNone(pca.q_residuals_)
        self.assertIsNotNone(pca.q_limit_)
        self.assertIsNotNone(pca.t_hotelling_)
        self.assertIsNotNone(pca.t_limit_)

    # PCA can transform a numpy array
    def test_transform_numpy_array(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        pca.fit(data)
        transformed_data = pca.transform(data)
        self.assertIsNotNone(transformed_data)

    # PCA can predict a numpy array
    def test_predict_numpy_array(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        pca.fit(data)
        predicted_data = pca.predict(data)
        self.assertIsNotNone(predicted_data)
        self.assertIsNotNone(pca.q_residuals_predicted_)
        self.assertIsNotNone(pca.t_hotelling_predicted_)

    # PCA can be fit, transformed, and predicted in sequence
    def test_fit_transform_predict_sequence(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        pca.fit(data)
        transformed_data = pca.transform(data)
        predicted_data = pca.predict(data)
        self.assertIsNotNone(transformed_data)
        self.assertIsNotNone(predicted_data)

    # PCA throws an error when fitting an array with less than 5 samples
    def test_fit_less_than_5_samples(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            pca.fit(data)

    # PCA throws an error when fitting an array with less features than n_components
    def test_fit_less_features_than_n_components(self):
        pca = PCA(n_components=3, mean_center=True)
        data = np.array([[1, 2], [4, 5], [7, 8], [10, 11], [13, 14]])
        with self.assertRaises(ValueError):
            pca.fit(data)

    # PCA throws a ValueError when calling the transform method before the fit method
    def test_transform_before_fit(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        with self.assertRaises(ValueError):
            pca.transform(data)

    # PCA throws a ValueError when calling the predict method before the fit method
    def test_predict_before_fit(self):
        pca = PCA(n_components=2, mean_center=True)
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                        [10, 11, 12], [13, 14, 15]])
        with self.assertRaises(ValueError):
            pca.predict(data)


if __name__ == '__main__':
    unittest.main()
