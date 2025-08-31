"""Model calibration and evaluation utilities."""

from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .schema import CalibrationMetrics, MatchPrediction


class ModelCalibrator:
    """Model calibration and reliability assessment."""
    
    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        """Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'platt')
            n_bins: Number of bins for calibration plots
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrators: Dict[str, any] = {}
        self.is_fitted = False
    
    def fit_calibration(
        self,
        predictions: List[MatchPrediction],
        actual_outcomes: List[str]
    ) -> None:
        """Fit calibration models on prediction-outcome pairs.
        
        Args:
            predictions: Model predictions
            actual_outcomes: Actual match outcomes ('H', 'D', 'A')
        """
        logger.info(f"Fitting calibration on {len(predictions)} predictions")
        
        # Convert to arrays
        pred_probs = np.array([
            [pred.prob_away_win, pred.prob_draw, pred.prob_home_win]
            for pred in predictions
        ])
        
        # Convert outcomes to numeric
        outcome_map = {'A': 0, 'D': 1, 'H': 2}
        y_true = np.array([outcome_map[outcome] for outcome in actual_outcomes])
        
        # Fit calibrator for each class
        for class_idx, class_name in enumerate(['away_win', 'draw', 'home_win']):
            y_binary = (y_true == class_idx).astype(int)
            probs = pred_probs[:, class_idx]
            
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:  # platt
                calibrator = LogisticRegression()
                probs = probs.reshape(-1, 1)
            
            calibrator.fit(probs, y_binary)
            self.calibrators[class_name] = calibrator
        
        self.is_fitted = True
        logger.info("Calibration models fitted successfully")
    
    def calibrate_predictions(
        self,
        predictions: List[MatchPrediction]
    ) -> List[MatchPrediction]:
        """Apply calibration to predictions.
        
        Args:
            predictions: Uncalibrated predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            logger.warning("Calibration not fitted. Returning original predictions.")
            return predictions
        
        calibrated_predictions = []
        
        for pred in predictions:
            # Extract probabilities
            probs = np.array([pred.prob_away_win, pred.prob_draw, pred.prob_home_win])
            
            # Apply calibration
            calibrated_probs = np.zeros(3)
            for class_idx, class_name in enumerate(['away_win', 'draw', 'home_win']):
                calibrator = self.calibrators[class_name]
                
                if self.method == "isotonic":
                    calibrated_probs[class_idx] = calibrator.predict([probs[class_idx]])[0]
                else:  # platt
                    calibrated_probs[class_idx] = calibrator.predict_proba(
                        probs[class_idx].reshape(1, -1)
                    )[0, 1]
            
            # Renormalize
            calibrated_probs /= calibrated_probs.sum()
            
            # Create calibrated prediction
            calibrated_pred = MatchPrediction(
                date=pred.date,
                home_team=pred.home_team,
                away_team=pred.away_team,
                prob_home_win=float(calibrated_probs[2]),
                prob_draw=float(calibrated_probs[1]),
                prob_away_win=float(calibrated_probs[0]),
                most_likely_score=pred.most_likely_score,
                most_likely_prob=pred.most_likely_prob,
                expected_home_goals=pred.expected_home_goals,
                expected_away_goals=pred.expected_away_goals,
                model_version=f"{pred.model_version}_calibrated",
                prediction_timestamp=pred.prediction_timestamp
            )
            
            calibrated_predictions.append(calibrated_pred)
        
        return calibrated_predictions
    
    def evaluate_calibration(
        self,
        predictions: List[MatchPrediction],
        actual_outcomes: List[str],
        model_name: str = "model"
    ) -> CalibrationMetrics:
        """Evaluate model calibration.
        
        Args:
            predictions: Model predictions
            actual_outcomes: Actual outcomes
            model_name: Name of the model being evaluated
            
        Returns:
            CalibrationMetrics object
        """
        logger.info(f"Evaluating calibration for {len(predictions)} predictions")
        
        # Convert to arrays
        pred_probs = np.array([
            [pred.prob_away_win, pred.prob_draw, pred.prob_home_win]
            for pred in predictions
        ])
        
        outcome_map = {'A': 0, 'D': 1, 'H': 2}
        y_true = np.array([outcome_map[outcome] for outcome in actual_outcomes])
        
        # Calculate overall metrics
        log_loss_score = log_loss(y_true, pred_probs)
        
        # Brier score (average across classes)
        brier_scores = []
        for class_idx in range(3):
            y_binary = (y_true == class_idx).astype(int)
            brier_scores.append(brier_score_loss(y_binary, pred_probs[:, class_idx]))
        avg_brier_score = np.mean(brier_scores)
        
        # Reliability curve for home win probability (most interpretable)
        home_win_probs = pred_probs[:, 2]
        home_win_actual = (y_true == 2).astype(int)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            home_win_actual, home_win_probs, n_bins=self.n_bins, strategy='quantile'
        )
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (home_win_probs > bin_lower) & (home_win_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = home_win_actual[in_bin].mean()
                avg_confidence_in_bin = home_win_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Get date range
        dates = [pred.date for pred in predictions]
        eval_start = min(dates)
        eval_end = max(dates)
        
        return CalibrationMetrics(
            model_name=model_name,
            brier_score=avg_brier_score,
            log_loss=log_loss_score,
            expected_calibration_error=ece,
            reliability_bins=mean_predicted_value.tolist(),
            reliability_frequencies=fraction_of_positives.tolist(),
            reliability_confidences=mean_predicted_value.tolist(),
            evaluation_start=eval_start,
            evaluation_end=eval_end,
            n_predictions=len(predictions)
        )
    
    def plot_reliability_diagram(
        self,
        predictions: List[MatchPrediction],
        actual_outcomes: List[str],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create reliability diagram.
        
        Args:
            predictions: Model predictions
            actual_outcomes: Actual outcomes
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to arrays
        pred_probs = np.array([pred.prob_home_win for pred in predictions])
        actual_home_wins = np.array([1 if outcome == 'H' else 0 for outcome in actual_outcomes])
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_home_wins, pred_probs, n_bins=self.n_bins, strategy='quantile'
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label='Model calibration')
        
        # Formatting
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Reliability Diagram - Home Win Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add calibration statistics
        ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) * 
                    (len(pred_probs) / self.n_bins)) / len(pred_probs)
        ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved reliability diagram to {save_path}")
        
        return fig
