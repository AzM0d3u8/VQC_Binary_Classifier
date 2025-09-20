import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import vqc_classifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vqc_classifier import QuantumMLPipeline


class TestQuantumMLPipeline(unittest.TestCase):
    """Comprehensive test suite for Quantum ML Pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = QuantumMLPipeline(max_iter=10, shots=512, random_seed=42)
        
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.max_iter, 10)
        self.assertEqual(self.pipeline.shots, 512)
        self.assertEqual(self.pipeline.random_seed, 42)
        self.assertIsInstance(self.pipeline.results, dict)
        
    def test_data_loading(self):
        """Test data loading and preprocessing"""
        X_train, X_test, y_train, y_test = self.pipeline.load_and_preprocess_data()
        
        # Check shapes and types
        self.assertEqual(X_train.shape[1], 2)  # 2 features
        self.assertEqual(X_test.shape[1], 2)
        self.assertEqual(len(set(y_train)), 2)  # Binary classification
        self.assertEqual(len(set(y_test)), 2)
        
        # Check data scaling (mean ≈ 0, std ≈ 1)
        self.assertAlmostEqual(np.mean(X_train), 0, places=1)
        self.assertAlmostEqual(np.std(X_train), 1, places=0)
        
        # Check train/test split ratio
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.3, places=1)
        
    def test_classical_baseline(self):
        """Test classical baseline training"""
        X_train, X_test, y_train, y_test = self.pipeline.load_and_preprocess_data()
        accuracy, predictions, training_time = self.pipeline.train_classical_baseline(
            X_train, y_train, X_test, y_test
        )
        
        # Check accuracy bounds
        self.assertTrue(0 <= accuracy <= 1)
        
        # Check predictions
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Check timing
        self.assertGreater(training_time, 0)
        
    @patch('vqc_classifier.VQC')
    @patch('vqc_classifier.Sampler')
    def test_vqc_training_mock(self, mock_sampler, mock_vqc):
        """Test VQC training with mocked Qiskit components"""
        # Mock VQC behavior
        mock_vqc_instance = MagicMock()
        mock_vqc_instance.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_vqc.return_value = mock_vqc_instance
        
        # Mock sampler
        mock_sampler_instance = MagicMock()
        mock_sampler.return_value = mock_sampler_instance
        
        X_train, X_test, y_train, y_test = self.pipeline.load_and_preprocess_data()
        
        # Limit test size for faster execution
        X_test_small = X_test[:5]
        y_test_small = y_test[:5]
        
        accuracy, predictions, training_time = self.pipeline.train_vqc(
            X_train, y_train, X_test_small, y_test_small
        )
        
        # Check that VQC was instantiated and used
        mock_vqc.assert_called_once()
        mock_vqc_instance.fit.assert_called_once()
        mock_vqc_instance.predict.assert_called_once()
        
        # Check results
        self.assertTrue(0 <= accuracy <= 1)
        self.assertEqual(len(predictions), len(y_test_small))
        
    def test_comprehensive_evaluation(self):
        """Test evaluation metrics generation"""
        # Create dummy data
        y_true = np.array([0, 0, 1, 1, 0, 1])
        vqc_pred = np.array([0, 1, 1, 1, 0, 0])
        classical_pred = np.array([0, 0, 1, 0, 0, 1])
        
        # Set required attributes
        self.pipeline.class_names = ['setosa', 'versicolor']
        
        metrics = self.pipeline.comprehensive_evaluation(y_true, vqc_pred, classical_pred)
        
        # Check structure
        self.assertIn('vqc', metrics)
        self.assertIn('classical', metrics)
        self.assertIn('report', metrics['vqc'])
        self.assertIn('confusion_matrix', metrics['vqc'])
        
        # Check confusion matrix shape
        self.assertEqual(metrics['vqc']['confusion_matrix'].shape, (2, 2))
        self.assertEqual(metrics['classical']['confusion_matrix'].shape, (2, 2))
        
    def test_results_structure(self):
        """Test results dictionary structure"""
        # Mock a complete pipeline run results
        mock_results = {
            'experiment_info': {
                'timestamp': '2024-09-21T14:30:45',
                'max_iterations': 10,
                'quantum_shots': 512
            },
            'performance_metrics': {
                'vqc_accuracy': 0.85,
                'classical_accuracy': 0.90,
                'quantum_advantage': -0.05
            }
        }
        
        # Check required keys exist
        self.assertIn('experiment_info', mock_results)
        self.assertIn('performance_metrics', mock_results)
        
        # Check metric types
        metrics = mock_results['performance_metrics']
        self.assertIsInstance(metrics['vqc_accuracy'], (int, float))
        self.assertIsInstance(metrics['classical_accuracy'], (int, float))
        self.assertIsInstance(metrics['quantum_advantage'], (int, float))
        
    def test_argument_parsing(self):
        """Test command line argument parsing"""
        from vqc_classifier import parse_arguments
        
        # Test with default arguments
        with patch('sys.argv', ['vqc_classifier.py']):
            args = parse_arguments()
            self.assertEqual(args.maxiter, 100)
            self.assertEqual(args.shots, 1024)
            self.assertEqual(args.seed, 42)
            
    def test_error_handling(self):
        """Test error handling in pipeline methods"""
        # Test with invalid data
        with self.assertRaises((ValueError, IndexError)):
            invalid_X = np.array([])
            invalid_y = np.array([])
            self.pipeline.train_classical_baseline(invalid_X, invalid_y, invalid_X, invalid_y)


class TestProjectStructure(unittest.TestCase):
    """Test project structure and file organization"""
    
    def test_required_files_exist(self):
        """Test that all required project files exist"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        required_files = [
            'vqc_classifier.py',
            'requirements.txt',
            'README.md',
            'config.json'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(base_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"Required file {file_name} not found")
            
    def test_config_json_structure(self):
        """Test configuration file structure"""
        import json
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'config.json')
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check required sections
        required_sections = ['experiment', 'quantum_settings', 'data_settings', 'output_settings']
        for section in required_sections:
            self.assertIn(section, config, f"Config section {section} missing")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)