import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import json
import time
from pathlib import Path
from datetime import datetime


class QuantumMLPipeline:
    """
    Professional Quantum Machine Learning Pipeline for Binary Classification
    (Classical simulation mode - for demonstration without quantum dependencies)
    """
    
    def __init__(self, max_iter=100, shots=1024, random_seed=42):
        """Initialize the Quantum ML Pipeline"""
        self.max_iter = max_iter
        self.shots = shots
        self.random_seed = random_seed
        self.results = {}
        self.setup_logging()
        self.logger.info("🚀 Quantum ML Pipeline initialized (Classical simulation mode)")
        
    def setup_logging(self):
        """Setup professional logging with file output"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / f"vqc_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self):
        """Load and preprocess Iris dataset for binary classification"""
        self.logger.info("📊 Loading and preprocessing Iris dataset...")
        
        # Load Iris dataset - using only 2 classes (Setosa vs Versicolor)
        iris = load_iris()
        X = iris.data[iris.target != 2][:, :2]  # Only first 2 features
        y = iris.target[iris.target != 2]
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, random_state=self.random_seed, test_size=0.3, stratify=y
        )
        
        self.logger.info(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Store for later use
        self.scaler = scaler
        self.feature_names = [iris.feature_names[i] for i in [0, 1]]
        self.class_names = [iris.target_names[i] for i in [0, 1]]
        
        return X_train, X_test, y_train, y_test

    def train_vqc_simulation(self, X_train, y_train, X_test, y_test):
        """
        Simulate VQC training using a classical approach
        (In real implementation, this would use Qiskit VQC)
        """
        self.logger.info("🔬 Training Variational Quantum Classifier (Simulation)...")
        start_time = time.time()
        
        # Simulate quantum-like behavior with a modified logistic regression
        # Add some noise to simulate quantum measurement uncertainty
        np.random.seed(self.random_seed)
        
        # Create a "quantum-inspired" classifier with slight randomness
        quantum_classifier = LogisticRegression(
            random_state=self.random_seed, 
            max_iter=self.max_iter,
            C=0.8  # Slightly different regularization to simulate quantum effects
        )
        quantum_classifier.fit(X_train, y_train)
        
        # Add measurement noise to simulate quantum uncertainty
        predictions_prob = quantum_classifier.predict_proba(X_test)
        noise = np.random.normal(0, 0.1, predictions_prob.shape)
        noisy_predictions_prob = predictions_prob + noise
        
        # Convert back to class predictions
        vqc_predictions = np.argmax(noisy_predictions_prob, axis=1)
        vqc_accuracy = accuracy_score(y_test, vqc_predictions)
        
        # Simulate quantum training time (slower than classical)
        time.sleep(1)  # Simulate quantum computation time
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ VQC training completed in {training_time:.2f}s")
        self.logger.info(f"🎯 VQC Accuracy: {vqc_accuracy:.4f} ({vqc_accuracy*100:.2f}%)")
        
        return vqc_accuracy, vqc_predictions, training_time

    def train_classical_baseline(self, X_train, y_train, X_test, y_test):
        """Train classical Logistic Regression baseline"""
        self.logger.info("🔬 Training Classical Baseline (Logistic Regression)...")
        start_time = time.time()
        
        classifier = LogisticRegression(random_state=self.random_seed, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ Classical training completed in {training_time:.4f}s")
        self.logger.info(f"🎯 Classical Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy, predictions, training_time

    def comprehensive_evaluation(self, y_true, vqc_pred, classical_pred):
        """Generate comprehensive evaluation metrics"""
        
        vqc_report = classification_report(y_true, vqc_pred, target_names=self.class_names, output_dict=True)
        classical_report = classification_report(y_true, classical_pred, target_names=self.class_names, output_dict=True)
        
        vqc_cm = confusion_matrix(y_true, vqc_pred)
        classical_cm = confusion_matrix(y_true, classical_pred)
        
        return {
            'vqc': {'report': vqc_report, 'confusion_matrix': vqc_cm},
            'classical': {'report': classical_report, 'confusion_matrix': classical_cm}
        }

    def create_visualizations(self, vqc_accuracy, classical_accuracy, vqc_time, classical_time, evaluation_metrics):
        """Create comprehensive visualizations"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧠 Quantum vs Classical ML Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        models = ['Quantum VQC\n(Simulated)', 'Classical\nLogistic Regression']
        accuracies = [vqc_accuracy, classical_accuracy]
        colors = ['#8A2BE2', '#4CAF50']
        
        bars = axes[0,0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        axes[0,0].set_title('🎯 Model Accuracy Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 2. Training time comparison
        times = [vqc_time, classical_time]
        bars = axes[0,1].bar(models, times, color=colors, alpha=0.8, edgecolor='black')
        axes[0,1].set_title('⏱️ Training Time Comparison', fontweight='bold')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, times):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02, 
                          f'{time_val:.3f}s', ha='center', fontweight='bold')
        
        # 3. VQC Confusion Matrix
        sns.heatmap(evaluation_metrics['vqc']['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[1,0].set_title('🔮 VQC Confusion Matrix', fontweight='bold')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # 4. Classical Confusion Matrix
        sns.heatmap(evaluation_metrics['classical']['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Greens', ax=axes[1,1],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[1,1].set_title('🧠 Classical Confusion Matrix', fontweight='bold')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plots
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(output_dir / f'quantum_ml_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        self.logger.info(f"📊 Visualization saved to results/quantum_ml_comparison_{timestamp}.png")
        plt.show()

    def save_results(self, results):
        """Save results to JSON file"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to results/results_{timestamp}.json")

    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        self.logger.info("🚀 Starting Quantum ML Pipeline...")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Train models
        vqc_accuracy, vqc_predictions, vqc_time = self.train_vqc_simulation(X_train, y_train, X_test, y_test)
        classical_accuracy, classical_predictions, classical_time = self.train_classical_baseline(X_train, y_train, X_test, y_test)
        
        # Comprehensive evaluation
        evaluation_metrics = self.comprehensive_evaluation(y_test, vqc_predictions, classical_predictions)
        
        # Create visualizations
        self.create_visualizations(vqc_accuracy, classical_accuracy, vqc_time, classical_time, evaluation_metrics)
        
        # Compile results
        results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'simulation',
                'max_iterations': self.max_iter,
                'quantum_shots': self.shots,
                'random_seed': self.random_seed,
                'dataset': 'Iris (Binary: Setosa vs Versicolor)',
                'features_used': self.feature_names
            },
            'performance_metrics': {
                'vqc_accuracy': float(vqc_accuracy),
                'classical_accuracy': float(classical_accuracy),
                'vqc_training_time': float(vqc_time),
                'classical_training_time': float(classical_time),
                'quantum_advantage': float(vqc_accuracy - classical_accuracy)
            },
            'detailed_evaluation': evaluation_metrics
        }
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results

    def print_summary(self, results):
        """Print a professional summary"""
        print("\n" + "="*80)
        print("🧠 QUANTUM MACHINE LEARNING PIPELINE - FINAL REPORT")
        print("="*80)
        
        metrics = results['performance_metrics']
        
        print(f"🎯 VQC Accuracy:         {metrics['vqc_accuracy']:.4f} ({metrics['vqc_accuracy']*100:.2f}%)")
        print(f"🧠 Classical Accuracy:   {metrics['classical_accuracy']:.4f} ({metrics['classical_accuracy']*100:.2f}%)")
        print(f"⚡ Quantum Advantage:    {metrics['quantum_advantage']:+.4f}")
        print(f"⏱️  VQC Training Time:    {metrics['vqc_training_time']:.4f}s")
        print(f"⏱️  Classical Time:       {metrics['classical_training_time']:.4f}s")
        
        if metrics['quantum_advantage'] > 0:
            print("🏆 Quantum model achieved superior performance!")
        else:
            print("📈 Classical model performed better (expected for this simple dataset)")
        
        print("\n📁 Results saved to:")
        print("   - ./results/quantum_ml_comparison_[timestamp].png")
        print("   - ./results/results_[timestamp].json")
        print("   - ./logs/vqc_pipeline_[timestamp].log")
        print("\n💡 Note: This demo uses classical simulation of quantum behavior.")
        print("    Install qiskit packages for real quantum implementation.")
        print("="*80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="🧠 Professional Quantum Machine Learning Pipeline (Demo Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vqc_demo.py                          # Default settings
  python vqc_demo.py --maxiter 200           # More optimization steps
  python vqc_demo.py --shots 2048            # More quantum shots
        """
    )
    
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations for optimizer (default: 100)")
    parser.add_argument("--shots", type=int, default=1024, help="Number of quantum circuit shots (default: 1024)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    pipeline = QuantumMLPipeline(
        max_iter=args.maxiter,
        shots=args.shots,
        random_seed=args.seed
    )
    
    try:
        results = pipeline.run_complete_pipeline()
        return 0
        
    except KeyboardInterrupt:
        pipeline.logger.info("🛑 Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        pipeline.logger.error(f"❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())