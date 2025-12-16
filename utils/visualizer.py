import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

class ResultsVisualizer:
    def __init__(self):
        plt.style.use('dark_background')
        self.cyber_colors = ['#00ff41', '#0080ff', '#bf00ff', '#ff0080']
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curves(self, results, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, result) in enumerate(results.items()):
            if 'probabilities' in result:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=self.cyber_colors[i % len(self.cyber_colors)],
                        label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - BlackWall Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()