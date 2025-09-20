# 🧠 Professional Quantum Machine Learning Pipeline

## Overview
A production-ready **Variational Quantum Classifier (VQC)** implementation that demonstrates quantum machine learning capabilities for binary classification. This project showcases modern quantum computing techniques using IBM Qiskit and compares quantum performance against classical baselines.

## 🏆 Key Features
- **Professional Architecture**: Object-oriented design with comprehensive error handling
- **Modern Qiskit Integration**: Uses latest Qiskit primitives and algorithms
- **Comprehensive Metrics**: Accuracy, confusion matrices, classification reports
- **Advanced Visualization**: Multi-panel comparison plots with professional styling
- **Robust Logging**: File-based logging with timestamps and structured output
- **Results Export**: JSON export of all metrics and experimental parameters
- **CLI Interface**: Command-line arguments for experiment configuration

## 🚀 Technologies Stack
- **Quantum Computing**: Qiskit, Qiskit Machine Learning, Qiskit Algorithms
- **Classical ML**: Scikit-learn, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas-style workflows with StandardScaler

## 📊 Dataset & Problem
- **Dataset**: Iris flower classification (binary subset)
- **Classes**: Setosa vs Versicolor (2-class problem)
- **Features**: Sepal length & width (2D for quantum circuit efficiency)
- **Preprocessing**: StandardScaler normalization for quantum compatibility

## 🔬 Quantum Architecture
- **Feature Map**: ZZFeatureMap with circular entanglement
- **Ansatz**: RealAmplitudes variational circuit
- **Optimizer**: COBYLA (Constrained Optimization BY Linear Approximations)
- **Backend**: Qiskit Aer simulator with configurable shot count

## 🛠️ Installation & Setup

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Quick Install
```bash
# Clone repository
git clone <your-repo-url>
cd VQC_Binary_Classifier

# Install dependencies
pip install -r requirements.txt
```

## 🚦 Usage Examples

### Basic Execution
```bash
python vqc_classifier.py
```

### Advanced Configuration
```bash
# High-precision run with more optimization steps
python vqc_classifier.py --maxiter 200 --shots 2048

# Quick test run
python vqc_classifier.py --maxiter 50 --shots 512

# Reproducible experiment with custom seed
python vqc_classifier.py --seed 123 --maxiter 150
```

### CLI Options
```
--maxiter    : Maximum COBYLA optimizer iterations (default: 100)
--shots      : Quantum circuit shots for sampling (default: 1024)  
--seed       : Random seed for reproducibility (default: 42)
```

## 📈 Expected Output

### Console Output
```
2024-09-21 14:30:15 [INFO] __main__: 🚀 Quantum ML Pipeline initialized
2024-09-21 14:30:15 [INFO] __main__: 📊 Loading and preprocessing Iris dataset...
2024-09-21 14:30:15 [INFO] __main__: ✅ Data loaded: 70 training samples, 30 test samples
2024-09-21 14:30:15 [INFO] __main__: 🔬 Training Variational Quantum Classifier...
2024-09-21 14:30:45 [INFO] __main__: ✅ VQC training completed in 29.84s
2024-09-21 14:30:45 [INFO] __main__: 🎯 VQC Accuracy: 0.9333 (93.33%)
2024-09-21 14:30:45 [INFO] __main__: 🔬 Training Classical Baseline...
2024-09-21 14:30:45 [INFO] __main__: ✅ Classical training completed in 0.0123s
2024-09-21 14:30:45 [INFO] __main__: 🎯 Classical Accuracy: 0.9667 (96.67%)

================================================================================
🧠 QUANTUM MACHINE LEARNING PIPELINE - FINAL REPORT
================================================================================
🎯 VQC Accuracy:         0.9333 (93.33%)
� Classical Accuracy:   0.9667 (96.67%)
⚡ Quantum Advantage:    -0.0334
⏱️  VQC Training Time:    29.8400s
⏱️  Classical Time:       0.0123s
📈 Classical model performed better (expected for this simple dataset)

📁 Results saved to:
   - ./results/quantum_ml_comparison_20240921_143045.png
   - ./results/results_20240921_143045.json
   - ./logs/vqc_pipeline_20240921_143045.log
================================================================================
```

### Generated Files
```
📁 VQC_Binary_Classifier/
├── results/
│   ├── quantum_ml_comparison_[timestamp].png    # 4-panel visualization
│   └── results_[timestamp].json                 # Complete metrics export
├── logs/
│   └── vqc_pipeline_[timestamp].log            # Detailed execution log
└── vqc_classifier.py                           # Main pipeline
```

## 📊 Visualization Output
The pipeline generates a comprehensive 4-panel visualization:
1. **Accuracy Comparison**: Bar chart comparing VQC vs Classical performance
2. **Training Time**: Performance timing comparison
3. **VQC Confusion Matrix**: Detailed classification breakdown
4. **Classical Confusion Matrix**: Baseline model performance

## 📋 Results Structure
```json
{
  "experiment_info": {
    "timestamp": "2024-09-21T14:30:45",
    "max_iterations": 100,
    "quantum_shots": 1024,
    "dataset": "Iris (Binary: Setosa vs Versicolor)",
    "features_used": ["sepal length", "sepal width"]
  },
  "performance_metrics": {
    "vqc_accuracy": 0.9333,
    "classical_accuracy": 0.9667,
    "quantum_advantage": -0.0334,
    "vqc_training_time": 29.84,
    "classical_training_time": 0.012
  }
}
```

## 🧪 Running Tests
```bash
python -m unittest test_vqc_classifier.py -v
```

## 🔧 Troubleshooting

### Common Issues
1. **ImportError**: Ensure all requirements are installed: `pip install -r requirements.txt`
2. **Memory Issues**: Reduce `--shots` parameter for systems with limited RAM
3. **Slow Training**: Decrease `--maxiter` for faster (less precise) results

### Performance Optimization
- **CPU**: VQC training is CPU-intensive; consider cloud platforms for production
- **Memory**: Each quantum shot requires memory; reduce shots if needed
- **Time**: Classical models train ~1000x faster; quantum advantage appears with larger datasets

## 🎯 Resume-Worthy Highlights
- **Quantum Computing**: Hands-on experience with variational quantum algorithms
- **Production Code**: Professional logging, error handling, CLI interfaces
- **ML Engineering**: Complete pipeline from data loading to results export
- **Modern Tools**: Latest Qiskit APIs, object-oriented design patterns
- **Visualization**: Professional-grade scientific plotting and reporting

## 📚 Technical Deep Dive
- **Quantum Circuit Depth**: 3-layer ansatz with 2-qubit ZZ feature mapping
- **Classical Baseline**: L-BFGS logistic regression with standard preprocessing
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrices
- **Reproducibility**: Seeded random number generation across quantum and classical components

## 🤝 Contributing
This project demonstrates quantum ML proficiency suitable for:
- Quantum computing research positions
- ML engineering roles requiring quantum knowledge
- Data science portfolios showcasing advanced techniques
- Academic or industry quantum algorithm development

---
**Author**: [Your Name]  
**Last Updated**: September 2024  
**Quantum Framework**: Qiskit >= 0.45.0

---

## 📚 Learnings
- Quantum circuits as parameterized models
- Hybrid quantum-classical optimization
- Basic quantum ML workflow with Qiskit
