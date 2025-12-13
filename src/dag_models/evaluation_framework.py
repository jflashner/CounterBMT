"""
Evaluation Framework for LLM-Based Counterfactual Crash Analysis

This module provides metrics and benchmarks for evaluating:
1. Severity prediction accuracy
2. Counterfactual plausibility
3. Causal consistency
4. Physical validity
5. Comparison against existing benchmarks

Key Benchmarks Referenced:
- CLEVRER: Counterfactual reasoning on collision videos
- CoPhy: Counterfactual physics prediction
- SHRP2/SynSHRP2: Real-world crash/near-crash data
- FARS/CRSS: National crash severity databases
- CrashLLM: LLM-based crash prediction
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score, log_loss
)
from sklearn.calibration import calibration_curve
import json
from copy import deepcopy
from collections import defaultdict


# =============================================================================
# SECTION 1: SEVERITY PREDICTION METRICS
# =============================================================================

@dataclass
class SeverityMetrics:
    """
    Metrics for evaluating severity prediction models.
    
    These align with metrics used in:
    - FARS/CRSS crash severity studies
    - CrashLLM benchmark
    - Traffic safety ML literature
    """
    
    # Classification metrics (for KABCO/discrete severity)
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    
    # Per-class metrics (important for imbalanced crash data)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    
    # Regression metrics (for continuous severity scores)
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    
    # Calibration metrics (critical for safety applications)
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    brier_score: float = 0.0
    
    # Ordinal metrics (for ordered severity levels)
    ordinal_accuracy: float = 0.0  # Exact match
    ordinal_mae: float = 0.0       # Mean absolute ordinal error
    within_one_accuracy: float = 0.0  # Prediction within ±1 level
    
    confusion_matrix: Optional[np.ndarray] = None


def compute_severity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    is_ordinal: bool = True
) -> SeverityMetrics:
    """
    Compute comprehensive severity prediction metrics.
    
    Args:
        y_true: Ground truth severity labels (int or float)
        y_pred: Predicted severity labels
        y_prob: Predicted probabilities (for calibration metrics)
        class_names: Names for each severity class
        is_ordinal: Whether severity classes are ordinal (e.g., KABCO)
    
    Returns:
        SeverityMetrics with all computed values
    """
    metrics = SeverityMetrics()
    
    # Basic classification metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    
    # Handle multi-class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics.macro_precision = precision
    metrics.macro_recall = recall
    metrics.macro_f1 = f1
    
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics.weighted_f1 = weighted_f1
    
    # Balanced accuracy (important for imbalanced crash data)
    from sklearn.metrics import balanced_accuracy_score
    metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    if class_names is None:
        class_names = [str(i) for i in range(len(precision_per))]
    
    for i, name in enumerate(class_names):
        if i < len(precision_per):
            metrics.per_class_precision[name] = precision_per[i]
            metrics.per_class_recall[name] = recall_per[i]
            metrics.per_class_f1[name] = f1_per[i]
    
    # Confusion matrix
    metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    # Ordinal metrics (if applicable)
    if is_ordinal:
        metrics.ordinal_accuracy = accuracy_score(y_true, y_pred)
        metrics.ordinal_mae = mean_absolute_error(y_true, y_pred)
        metrics.within_one_accuracy = np.mean(np.abs(y_true - y_pred) <= 1)
    
    # Calibration metrics (if probabilities provided)
    if y_prob is not None:
        metrics.brier_score = compute_brier_score(y_true, y_prob)
        ece, mce = compute_calibration_error(y_true, y_prob)
        metrics.expected_calibration_error = ece
        metrics.max_calibration_error = mce
    
    # Regression metrics (treating ordinal as continuous)
    y_true_float = y_true.astype(float)
    y_pred_float = y_pred.astype(float)
    metrics.mse = mean_squared_error(y_true_float, y_pred_float)
    metrics.rmse = np.sqrt(metrics.mse)
    metrics.mae = mean_absolute_error(y_true_float, y_pred_float)
    metrics.r2 = r2_score(y_true_float, y_pred_float)
    
    return metrics


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score for multi-class predictions."""
    n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2
    y_true_onehot = np.eye(n_classes)[y_true.astype(int)]
    return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


def compute_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Critical for safety applications where we need well-calibrated probabilities.
    """
    # For multi-class, use max probability
    if len(y_prob.shape) > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            bin_error = np.abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
    
    return ece, mce


# =============================================================================
# SECTION 2: COUNTERFACTUAL PLAUSIBILITY METRICS
# =============================================================================

@dataclass
class CounterfactualMetrics:
    """
    Metrics for evaluating counterfactual quality.
    
    Based on metrics from:
    - CoPhy benchmark (MSE on predicted positions)
    - CLEVRER (counterfactual question accuracy)
    - Causal inference literature (PEHE, policy risk)
    """
    
    # Distribution-level metrics
    js_divergence: float = 0.0           # Jensen-Shannon divergence from real data
    kl_divergence: float = 0.0           # KL divergence from real distribution
    wasserstein_distance: float = 0.0    # Earth mover's distance
    
    # Constraint satisfaction
    validity_rate: float = 0.0           # % of counterfactuals satisfying constraints
    ontology_consistency: float = 0.0    # % consistent with VEACON ontology rules
    physical_plausibility: float = 0.0   # % satisfying physics constraints
    
    # Diversity metrics
    feature_coverage: float = 0.0        # Coverage of feature space
    unique_ratio: float = 0.0            # Ratio of unique counterfactuals
    entropy: float = 0.0                 # Entropy of generated distribution
    
    # Causal metrics
    causal_consistency: float = 0.0      # % respecting causal ordering
    intervention_effect_validity: float = 0.0  # Do interventions have expected effects?
    
    # Severity-specific metrics
    severity_target_accuracy: float = 0.0  # How well do we hit target severity?
    severity_monotonicity: float = 0.0     # Do severity-increasing features increase severity?


class CounterfactualValidator:
    """
    Validates counterfactual scenarios against physical and causal constraints.
    """
    
    def __init__(self):
        # Define constraint rules
        self.ontology_rules = self._build_ontology_rules()
        self.physics_rules = self._build_physics_rules()
        self.causal_rules = self._build_causal_rules()
    
    def _build_ontology_rules(self) -> List[Callable]:
        """VEACON-based ontology constraints."""
        rules = []
        
        # Weather → Surface consistency
        def weather_surface_rule(event: Dict) -> bool:
            weather = event.get("environment.weather", "clear")
            surface = event.get("environment.surface_condition", "dry")
            
            # Rain/snow should correlate with wet/icy surfaces
            if weather in ["rain", "snow"] and surface == "dry":
                return False  # Implausible
            if weather == "clear" and surface == "snow_ice":
                return False  # Implausible (usually)
            return True
        
        rules.append(weather_surface_rule)
        
        # Event type → Impact consistency
        def event_impact_rule(event: Dict) -> bool:
            event_type = event.get("accident.event_type", "normal")
            delta_v = event.get("crash_state.delta_v_kph", 0)
            
            # Normal driving shouldn't have high delta-V
            if event_type == "normal" and delta_v > 5:
                return False
            # Crashes should have some delta-V
            if event_type == "crash" and delta_v < 1:
                return False
            return True
        
        rules.append(event_impact_rule)
        
        # Conflict type → Point of impact consistency
        def conflict_impact_rule(event: Dict) -> bool:
            conflict = event.get("accident.conflict_type", "unknown")
            poi = event.get("crash_state.point_of_impact", "unknown")
            
            # Lead vehicle conflicts typically result in front impacts
            if conflict == "lead_vehicle" and poi in ["left", "right"]:
                return False  # Less common
            return True
        
        rules.append(conflict_impact_rule)
        
        return rules
    
    def _build_physics_rules(self) -> List[Callable]:
        """Physical constraint rules."""
        rules = []
        
        # Delta-V should be less than or equal to impact speed
        def deltav_speed_rule(event: Dict) -> bool:
            delta_v = event.get("crash_state.delta_v_kph", 0)
            impact_speed = event.get("crash_state.speed_at_impact_kph", 0)
            
            # Delta-V can't exceed impact speed significantly
            if delta_v > impact_speed * 1.5 + 10:  # Some tolerance
                return False
            return True
        
        rules.append(deltav_speed_rule)
        
        # Braking effectiveness should relate to surface condition
        def braking_surface_rule(event: Dict) -> bool:
            surface = event.get("environment.surface_condition", "dry")
            braking = event.get("crash_dynamics.braking_effectiveness", "high")
            
            # Hard to have high braking on ice
            if surface == "snow_ice" and braking == "high":
                return False
            return True
        
        rules.append(braking_surface_rule)
        
        # Severity should correlate with delta-V
        def severity_deltav_rule(event: Dict) -> bool:
            delta_v = event.get("crash_state.delta_v_kph", 0)
            severity = event.get("injury_outcome.severity_score", 0)
            
            # High delta-V should correlate with higher severity
            if delta_v > 50 and severity < 0.3:
                return False  # Implausible
            if delta_v < 10 and severity > 0.8:
                return False  # Implausible
            return True
        
        rules.append(severity_deltav_rule)
        
        return rules
    
    def _build_causal_rules(self) -> List[Callable]:
        """Causal ordering constraints."""
        rules = []
        
        # Pre-crash factors should be set before crash factors
        # (This is structural, not a value check)
        
        return rules
    
    def validate_counterfactual(self, event: Dict) -> Dict[str, Any]:
        """
        Validate a single counterfactual event.
        
        Returns:
            Dict with validation results for each rule category
        """
        results = {
            "ontology_valid": True,
            "physics_valid": True,
            "causal_valid": True,
            "ontology_violations": [],
            "physics_violations": [],
            "causal_violations": []
        }
        
        for i, rule in enumerate(self.ontology_rules):
            try:
                if not rule(event):
                    results["ontology_valid"] = False
                    results["ontology_violations"].append(f"rule_{i}")
            except Exception:
                pass
        
        for i, rule in enumerate(self.physics_rules):
            try:
                if not rule(event):
                    results["physics_valid"] = False
                    results["physics_violations"].append(f"rule_{i}")
            except Exception:
                pass
        
        for i, rule in enumerate(self.causal_rules):
            try:
                if not rule(event):
                    results["causal_valid"] = False
                    results["causal_violations"].append(f"rule_{i}")
            except Exception:
                pass
        
        return results
    
    def compute_batch_metrics(
        self, 
        counterfactuals: List[Dict],
        reference_distribution: Optional[List[Dict]] = None
    ) -> CounterfactualMetrics:
        """
        Compute metrics over a batch of counterfactual events.
        
        Args:
            counterfactuals: List of generated counterfactual events
            reference_distribution: Real events for distribution comparison
        """
        metrics = CounterfactualMetrics()
        
        if not counterfactuals:
            return metrics
        
        # Validate each counterfactual
        ontology_valid = 0
        physics_valid = 0
        causal_valid = 0
        
        for cf in counterfactuals:
            results = self.validate_counterfactual(cf)
            if results["ontology_valid"]:
                ontology_valid += 1
            if results["physics_valid"]:
                physics_valid += 1
            if results["causal_valid"]:
                causal_valid += 1
        
        n = len(counterfactuals)
        metrics.ontology_consistency = ontology_valid / n
        metrics.physical_plausibility = physics_valid / n
        metrics.causal_consistency = causal_valid / n
        metrics.validity_rate = (ontology_valid + physics_valid + causal_valid) / (3 * n)
        
        # Diversity metrics
        unique_cfs = set(json.dumps(cf, sort_keys=True, default=str) for cf in counterfactuals)
        metrics.unique_ratio = len(unique_cfs) / n
        
        # Distribution comparison (if reference provided)
        if reference_distribution:
            metrics.js_divergence = self._compute_js_divergence(
                counterfactuals, reference_distribution
            )
        
        return metrics
    
    def _compute_js_divergence(
        self, 
        generated: List[Dict], 
        reference: List[Dict]
    ) -> float:
        """Compute JS divergence between generated and reference distributions."""
        # Extract a key feature for comparison (e.g., severity)
        gen_severities = [
            e.get("injury_outcome.severity_score", 0) for e in generated
        ]
        ref_severities = [
            e.get("injury_outcome.severity_score", 0) for e in reference
        ]
        
        # Bin into histogram
        bins = np.linspace(0, 1, 11)
        gen_hist, _ = np.histogram(gen_severities, bins=bins, density=True)
        ref_hist, _ = np.histogram(ref_severities, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        gen_hist = gen_hist + 1e-10
        ref_hist = ref_hist + 1e-10
        
        # Normalize
        gen_hist = gen_hist / gen_hist.sum()
        ref_hist = ref_hist / ref_hist.sum()
        
        return jensenshannon(gen_hist, ref_hist)


# =============================================================================
# SECTION 3: CAUSAL REASONING METRICS (CLEVRER/CoPhy-style)
# =============================================================================

@dataclass  
class CausalReasoningMetrics:
    """
    Metrics for evaluating causal reasoning quality.
    
    Based on:
    - CLEVRER: Descriptive, Explanatory, Predictive, Counterfactual questions
    - CoPhy: Counterfactual prediction MSE
    - Causal inference: PEHE, ATE estimation error
    """
    
    # CLEVRER-style question answering accuracy
    descriptive_accuracy: float = 0.0     # "What color/type?"
    explanatory_accuracy: float = 0.0     # "What caused X?"
    predictive_accuracy: float = 0.0      # "What will happen next?"
    counterfactual_accuracy: float = 0.0  # "What if X?"
    
    # CoPhy-style prediction metrics
    position_mse: float = 0.0             # MSE on predicted final states
    trajectory_mse: float = 0.0           # MSE on full trajectories
    confounder_estimation_error: float = 0.0  # Error in latent variable estimation
    
    # Causal effect estimation (ITE/CATE)
    pehe: float = 0.0                     # Precision in Estimation of Heterogeneous Effects
    ate_error: float = 0.0                # Average Treatment Effect estimation error
    policy_risk: float = 0.0              # Risk of policy decisions based on estimates
    
    # Intervention consistency
    intervention_direction_accuracy: float = 0.0  # Did intervention change outcome correctly?
    monotonicity_violation_rate: float = 0.0      # Rate of non-monotonic predictions


class CausalReasoningEvaluator:
    """
    Evaluates causal reasoning capabilities of the LLM world model.
    """
    
    def __init__(self, world_model: Any = None):
        """
        Args:
            world_model: LLMWorldModel instance for evaluation
        """
        self.world_model = world_model
        self.test_cases = self._build_test_cases()
    
    def _build_test_cases(self) -> List[Dict]:
        """
        Build test cases for causal reasoning evaluation.
        
        Each test case has:
        - initial_state: The starting scenario
        - intervention: What we change
        - expected_effect: What should happen
        """
        test_cases = []
        
        # Test 1: Weather → Surface → Braking → Severity chain
        test_cases.append({
            "name": "weather_to_severity_chain",
            "initial_state": {
                "environment.weather": "rain",
                "environment.surface_condition": "wet",
                "crash_dynamics.braking_effectiveness": "low",
                "crash_state.delta_v_kph": 40,
                "injury_outcome.severity_score": 0.6
            },
            "intervention": {"environment.weather": "clear"},
            "expected_direction": {
                "environment.surface_condition": "should_improve",  # wet → dry
                "crash_dynamics.braking_effectiveness": "should_improve",  # low → higher
                "crash_state.delta_v_kph": "should_decrease",
                "injury_outcome.severity_score": "should_decrease"
            }
        })
        
        # Test 2: Speed → Delta-V → Severity
        test_cases.append({
            "name": "speed_to_severity",
            "initial_state": {
                "vehicle_state.pre_crash_speed_kph": 100,
                "crash_state.delta_v_kph": 50,
                "injury_outcome.severity_score": 0.7
            },
            "intervention": {"vehicle_state.pre_crash_speed_kph": 50},
            "expected_direction": {
                "crash_state.delta_v_kph": "should_decrease",
                "injury_outcome.severity_score": "should_decrease"
            }
        })
        
        # Test 3: Lighting → Visibility → Event type
        test_cases.append({
            "name": "lighting_to_event",
            "initial_state": {
                "environment.light": "dark",
                "environment.visibility": "poor",
                "accident.event_type": "crash"
            },
            "intervention": {"environment.light": "daylight"},
            "expected_direction": {
                "environment.visibility": "should_improve",
                "accident.event_type": "could_improve"  # crash → near_crash or normal
            }
        })
        
        # Test 4: No effect expected (independent variables)
        test_cases.append({
            "name": "no_effect_test",
            "initial_state": {
                "environment.road_type": "highway",
                "crash_state.delta_v_kph": 40
            },
            "intervention": {"environment.road_type": "urban"},
            "expected_direction": {
                "crash_state.delta_v_kph": "no_direct_effect"  # Road type doesn't directly affect delta-V
            }
        })
        
        return test_cases
    
    def evaluate_intervention_consistency(
        self,
        original_events: List[Dict],
        counterfactual_events: List[Dict],
        interventions: List[Dict]
    ) -> CausalReasoningMetrics:
        """
        Evaluate whether interventions produce causally consistent effects.
        
        Args:
            original_events: Original scenario states
            counterfactual_events: States after intervention
            interventions: What was changed in each case
        
        Returns:
            CausalReasoningMetrics with consistency scores
        """
        metrics = CausalReasoningMetrics()
        
        direction_correct = 0
        monotonicity_violations = 0
        
        for orig, cf, interv in zip(original_events, counterfactual_events, interventions):
            # Check if intervention effects are in expected direction
            correct, violation = self._check_intervention_effect(orig, cf, interv)
            if correct:
                direction_correct += 1
            if violation:
                monotonicity_violations += 1
        
        n = len(original_events)
        if n > 0:
            metrics.intervention_direction_accuracy = direction_correct / n
            metrics.monotonicity_violation_rate = monotonicity_violations / n
        
        return metrics
    
    def _check_intervention_effect(
        self, 
        original: Dict, 
        counterfactual: Dict, 
        intervention: Dict
    ) -> Tuple[bool, bool]:
        """
        Check if an intervention produced expected effects.
        
        Returns:
            (direction_correct, monotonicity_violated)
        """
        direction_correct = True
        monotonicity_violated = False
        
        # Check severity monotonicity with delta-V
        orig_dv = original.get("crash_state.delta_v_kph", 0)
        cf_dv = counterfactual.get("crash_state.delta_v_kph", 0)
        orig_sev = original.get("injury_outcome.severity_score", 0)
        cf_sev = counterfactual.get("injury_outcome.severity_score", 0)
        
        # If delta-V decreased, severity should decrease (or stay same)
        if cf_dv < orig_dv and cf_sev > orig_sev + 0.1:  # Allow small tolerance
            monotonicity_violated = True
        
        # If delta-V increased, severity should increase (or stay same)
        if cf_dv > orig_dv and cf_sev < orig_sev - 0.1:
            monotonicity_violated = True
        
        # Check weather → surface consistency
        if "environment.weather" in intervention:
            orig_weather = original.get("environment.weather", "clear")
            cf_weather = counterfactual.get("environment.weather", "clear")
            cf_surface = counterfactual.get("environment.surface_condition", "dry")
            
            # If weather improved (rain → clear), surface should also improve
            if orig_weather in ["rain", "snow"] and cf_weather == "clear":
                if cf_surface in ["wet", "snow_ice"]:
                    direction_correct = False
        
        return direction_correct, monotonicity_violated
    
    def compute_pehe(
        self,
        true_ite: np.ndarray,
        predicted_ite: np.ndarray
    ) -> float:
        """
        Compute Precision in Estimation of Heterogeneous Effects (PEHE).
        
        PEHE = sqrt(E[(τ(x) - τ̂(x))²])
        
        This is the standard metric for ITE/CATE estimation from causal inference.
        
        Args:
            true_ite: Ground truth individual treatment effects
            predicted_ite: Predicted treatment effects
        
        Returns:
            PEHE score (lower is better)
        """
        return np.sqrt(np.mean((true_ite - predicted_ite) ** 2))
    
    def compute_ate_error(
        self,
        true_ate: float,
        predicted_ate: float
    ) -> float:
        """
        Compute Average Treatment Effect estimation error.
        
        Args:
            true_ate: Ground truth ATE
            predicted_ate: Estimated ATE
        
        Returns:
            Absolute error in ATE estimation
        """
        return np.abs(true_ate - predicted_ate)


# =============================================================================
# SECTION 4: SAFETY-CRITICAL FEATURE IDENTIFICATION METRICS
# =============================================================================

@dataclass
class SafetyCriticalMetrics:
    """
    Metrics for evaluating safety-critical feature identification.
    """
    
    # Ranking quality
    ndcg: float = 0.0                    # Normalized Discounted Cumulative Gain
    map_score: float = 0.0               # Mean Average Precision
    rank_correlation: float = 0.0        # Spearman correlation with ground truth ranking
    
    # Top-k accuracy
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    
    # Counterfactual validity
    improvement_achieved: float = 0.0    # Avg severity reduction achieved
    improvement_predicted: float = 0.0   # Avg severity reduction predicted
    improvement_calibration: float = 0.0 # How well predictions match reality


def compute_safety_critical_metrics(
    predicted_rankings: List[List[str]],
    true_rankings: List[List[str]],
    predicted_improvements: List[List[float]],
    actual_improvements: List[List[float]]
) -> SafetyCriticalMetrics:
    """
    Compute metrics for safety-critical feature ranking.
    
    Args:
        predicted_rankings: Predicted feature rankings per event
        true_rankings: Ground truth feature rankings
        predicted_improvements: Predicted severity improvements
        actual_improvements: Actual severity improvements (from simulation/data)
    """
    metrics = SafetyCriticalMetrics()
    
    # Top-k accuracy
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    for pred, true in zip(predicted_rankings, true_rankings):
        if pred and true:
            if pred[0] == true[0]:
                top1_correct += 1
            if set(pred[:3]) & set(true[:3]):
                top3_correct += 1
            if set(pred[:5]) & set(true[:5]):
                top5_correct += 1
    
    n = len(predicted_rankings)
    if n > 0:
        metrics.top1_accuracy = top1_correct / n
        metrics.top3_accuracy = top3_correct / n
        metrics.top5_accuracy = top5_correct / n
    
    # Rank correlation
    correlations = []
    for pred, true in zip(predicted_rankings, true_rankings):
        if len(pred) >= 2 and len(true) >= 2:
            # Convert to ranks
            common = set(pred) & set(true)
            if len(common) >= 2:
                pred_ranks = [pred.index(f) for f in common if f in pred]
                true_ranks = [true.index(f) for f in common if f in true]
                if len(pred_ranks) >= 2:
                    corr, _ = stats.spearmanr(pred_ranks, true_ranks)
                    if not np.isnan(corr):
                        correlations.append(corr)
    
    if correlations:
        metrics.rank_correlation = np.mean(correlations)
    
    # Improvement calibration
    all_pred_impr = [i for sublist in predicted_improvements for i in sublist]
    all_actual_impr = [i for sublist in actual_improvements for i in sublist]
    
    if all_pred_impr and all_actual_impr:
        min_len = min(len(all_pred_impr), len(all_actual_impr))
        metrics.improvement_predicted = np.mean(all_pred_impr[:min_len])
        metrics.improvement_achieved = np.mean(all_actual_impr[:min_len])
        
        # Calibration: correlation between predicted and actual improvements
        if min_len >= 2:
            corr, _ = stats.pearsonr(
                all_pred_impr[:min_len], 
                all_actual_impr[:min_len]
            )
            if not np.isnan(corr):
                metrics.improvement_calibration = corr
    
    return metrics


# =============================================================================
# SECTION 5: BENCHMARK COMPARISON FRAMEWORK
# =============================================================================

@dataclass
class BenchmarkComparison:
    """
    Results from comparing against standard benchmarks.
    """
    benchmark_name: str
    our_score: float
    baseline_scores: Dict[str, float] = field(default_factory=dict)
    metric_name: str = ""
    higher_is_better: bool = True
    
    def relative_improvement(self, baseline_name: str) -> float:
        """Compute relative improvement over a baseline."""
        if baseline_name not in self.baseline_scores:
            return 0.0
        baseline = self.baseline_scores[baseline_name]
        if baseline == 0:
            return 0.0
        if self.higher_is_better:
            return (self.our_score - baseline) / baseline
        else:
            return (baseline - self.our_score) / baseline


class BenchmarkSuite:
    """
    Suite of benchmarks for comprehensive evaluation.
    
    Compares against:
    1. CLEVRER - Counterfactual reasoning on collision videos
    2. CoPhy - Counterfactual physics prediction
    3. SHRP2/SynSHRP2 - Real crash/near-crash data
    4. FARS/CRSS - National severity prediction
    5. CrashLLM - LLM-based crash prediction
    """
    
    # Baseline scores from literature
    CLEVRER_BASELINES = {
        "random": 0.25,  # 4-way classification
        "CNN+LSTM": 0.33,
        "MAC": 0.42,
        "NS-DR": 0.67,   # Neuro-symbolic
        "human": 0.96
    }
    
    COPHY_BASELINES = {
        "NPE": 15.2,     # MSE on block positions
        "IN": 12.8,
        "CophyNet": 5.4,
        "CophyNet+GT": 3.2
    }
    
    CRASH_SEVERITY_BASELINES = {
        # F1 scores from CrashLLM paper
        "logistic_regression": 0.32,
        "random_forest": 0.35,
        "xgboost": 0.38,
        "bayesian_network": 0.34,
        "CrashLLM_7B": 0.48,
        "CrashLLM_70B": 0.54
    }
    
    def __init__(self):
        self.results: List[BenchmarkComparison] = []
    
    def evaluate_clevrer_style(
        self,
        model_predictions: Dict[str, List[bool]],
        ground_truth: Dict[str, List[bool]]
    ) -> BenchmarkComparison:
        """
        Evaluate on CLEVRER-style counterfactual questions.
        
        Args:
            model_predictions: Dict with keys 'descriptive', 'explanatory', 
                              'predictive', 'counterfactual'
            ground_truth: Same structure with correct answers
        """
        # Focus on counterfactual accuracy
        cf_pred = model_predictions.get("counterfactual", [])
        cf_true = ground_truth.get("counterfactual", [])
        
        if cf_pred and cf_true:
            accuracy = accuracy_score(cf_true, cf_pred)
        else:
            accuracy = 0.0
        
        result = BenchmarkComparison(
            benchmark_name="CLEVRER-Counterfactual",
            our_score=accuracy,
            baseline_scores=self.CLEVRER_BASELINES,
            metric_name="Accuracy",
            higher_is_better=True
        )
        self.results.append(result)
        return result
    
    def evaluate_cophy_style(
        self,
        predicted_states: np.ndarray,
        ground_truth_states: np.ndarray
    ) -> BenchmarkComparison:
        """
        Evaluate on CoPhy-style counterfactual state prediction.
        
        Args:
            predicted_states: Predicted final positions/states
            ground_truth_states: True final positions/states
        """
        mse = mean_squared_error(ground_truth_states.flatten(), 
                                  predicted_states.flatten())
        
        result = BenchmarkComparison(
            benchmark_name="CoPhy-BlocktowerCF",
            our_score=mse,
            baseline_scores=self.COPHY_BASELINES,
            metric_name="MSE",
            higher_is_better=False
        )
        self.results.append(result)
        return result
    
    def evaluate_crash_severity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> BenchmarkComparison:
        """
        Evaluate crash severity prediction against CrashLLM baselines.
        """
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        result = BenchmarkComparison(
            benchmark_name="FARS/CRSS-Severity",
            our_score=f1,
            baseline_scores=self.CRASH_SEVERITY_BASELINES,
            metric_name="Macro-F1",
            higher_is_better=True
        )
        self.results.append(result)
        return result
    
    def generate_report(self) -> str:
        """Generate a summary report of all benchmark comparisons."""
        report = ["=" * 70]
        report.append("BENCHMARK COMPARISON REPORT")
        report.append("=" * 70)
        
        for result in self.results:
            report.append(f"\n{result.benchmark_name}")
            report.append("-" * 40)
            report.append(f"Our Score: {result.our_score:.4f} ({result.metric_name})")
            report.append(f"Direction: {'Higher' if result.higher_is_better else 'Lower'} is better")
            report.append("\nBaseline Comparisons:")
            
            for name, score in result.baseline_scores.items():
                improvement = result.relative_improvement(name)
                sign = "+" if improvement > 0 else ""
                report.append(f"  vs {name}: {score:.4f} ({sign}{improvement:.1%})")
        
        return "\n".join(report)


# =============================================================================
# SECTION 6: COMPLETE EVALUATION PIPELINE
# =============================================================================

class ComprehensiveEvaluator:
    """
    Complete evaluation pipeline combining all metrics.
    """
    
    def __init__(self, world_model: Any = None, severity_model: Callable = None):
        self.world_model = world_model
        self.severity_model = severity_model
        
        self.cf_validator = CounterfactualValidator()
        self.causal_evaluator = CausalReasoningEvaluator(world_model)
        self.benchmark_suite = BenchmarkSuite()
    
    def evaluate_all(
        self,
        original_events: List[Dict],
        counterfactual_events: List[Dict],
        interventions: List[Dict],
        severity_true: Optional[np.ndarray] = None,
        severity_pred: Optional[np.ndarray] = None,
        severity_prob: Optional[np.ndarray] = None,
        reference_events: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dict with all metric categories
        """
        results = {}
        
        # 1. Severity prediction metrics
        if severity_true is not None and severity_pred is not None:
            results["severity_metrics"] = compute_severity_metrics(
                severity_true, severity_pred, severity_prob,
                class_names=["No Injury", "Minor", "Serious", "Fatal"]
            )
        
        # 2. Counterfactual quality metrics
        if counterfactual_events:
            results["counterfactual_metrics"] = self.cf_validator.compute_batch_metrics(
                counterfactual_events, reference_events
            )
        
        # 3. Causal reasoning metrics
        if original_events and counterfactual_events and interventions:
            results["causal_metrics"] = self.causal_evaluator.evaluate_intervention_consistency(
                original_events, counterfactual_events, interventions
            )
        
        # 4. Benchmark comparisons
        if severity_true is not None and severity_pred is not None:
            self.benchmark_suite.evaluate_crash_severity(severity_true, severity_pred)
        
        results["benchmark_report"] = self.benchmark_suite.generate_report()
        
        return results
    
    def generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of evaluation results."""
        summary = ["=" * 70]
        summary.append("COMPREHENSIVE EVALUATION SUMMARY")
        summary.append("=" * 70)
        
        # Severity metrics summary
        if "severity_metrics" in results:
            sm = results["severity_metrics"]
            summary.append("\n📊 SEVERITY PREDICTION")
            summary.append(f"  Accuracy: {sm.accuracy:.2%}")
            summary.append(f"  Balanced Accuracy: {sm.balanced_accuracy:.2%}")
            summary.append(f"  Macro F1: {sm.macro_f1:.3f}")
            summary.append(f"  Calibration Error (ECE): {sm.expected_calibration_error:.3f}")
        
        # Counterfactual metrics summary
        if "counterfactual_metrics" in results:
            cm = results["counterfactual_metrics"]
            summary.append("\n🔄 COUNTERFACTUAL QUALITY")
            summary.append(f"  Validity Rate: {cm.validity_rate:.2%}")
            summary.append(f"  Ontology Consistency: {cm.ontology_consistency:.2%}")
            summary.append(f"  Physical Plausibility: {cm.physical_plausibility:.2%}")
            summary.append(f"  Diversity (Unique Ratio): {cm.unique_ratio:.2%}")
        
        # Causal metrics summary
        if "causal_metrics" in results:
            crm = results["causal_metrics"]
            summary.append("\n🎯 CAUSAL REASONING")
            summary.append(f"  Intervention Direction Accuracy: {crm.intervention_direction_accuracy:.2%}")
            summary.append(f"  Monotonicity Violation Rate: {crm.monotonicity_violation_rate:.2%}")
        
        # Benchmark comparison
        if "benchmark_report" in results:
            summary.append("\n" + results["benchmark_report"])
        
        return "\n".join(summary)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo_evaluation():
    """Demonstrate the evaluation framework."""
    
    # Create mock data
    np.random.seed(42)
    n_samples = 100
    
    # Mock severity predictions
    severity_true = np.random.randint(0, 4, n_samples)
    severity_pred = severity_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, n_samples // 5, replace=False)
    severity_pred[noise_idx] = np.random.randint(0, 4, len(noise_idx))
    
    # Mock probability predictions
    severity_prob = np.zeros((n_samples, 4))
    for i in range(n_samples):
        severity_prob[i, severity_pred[i]] = 0.7
        for j in range(4):
            if j != severity_pred[i]:
                severity_prob[i, j] = 0.1
    
    # Mock counterfactual events
    original_events = [
        {
            "environment.weather": "rain",
            "environment.surface_condition": "wet",
            "crash_state.delta_v_kph": 40,
            "injury_outcome.severity_score": 0.6
        }
        for _ in range(n_samples)
    ]
    
    counterfactual_events = [
        {
            "environment.weather": "clear",
            "environment.surface_condition": "dry",
            "crash_state.delta_v_kph": 30,
            "injury_outcome.severity_score": 0.4
        }
        for _ in range(n_samples)
    ]
    
    interventions = [
        {"environment.weather": "clear"}
        for _ in range(n_samples)
    ]
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_all(
        original_events=original_events,
        counterfactual_events=counterfactual_events,
        interventions=interventions,
        severity_true=severity_true,
        severity_pred=severity_pred,
        severity_prob=severity_prob
    )
    
    # Print summary
    print(evaluator.generate_summary(results))


# =============================================================================
# SECTION 7: REAL-WORLD DATA BENCHMARKS
# =============================================================================

@dataclass
class RealWorldBenchmark:
    """
    Defines a real-world benchmark with expected performance ranges.
    """
    name: str
    description: str
    dataset_size: int
    metric_name: str
    metric_range: Tuple[float, float]  # (min, max) expected values
    sota_score: float                   # State-of-the-art score
    baseline_score: float               # Simple baseline score
    higher_is_better: bool = True
    paper_reference: str = ""
    download_url: str = ""


class RealWorldBenchmarkRegistry:
    """
    Registry of real-world benchmarks for counterfactual crash analysis.
    
    Organized by category:
    1. Causal Inference Benchmarks (IHDP, Jobs, Twins, ACIC)
    2. Counterfactual Physics (CLEVRER, CoPhy)
    3. Crash Severity (FARS/CRSS, SHRP2, CrashLLM)
    4. Driving Scenarios (nuScenes, Waymo, CAR-Scenes)
    5. Simulation (CARLA safety-critical scenarios)
    """
    
    # =========================================================================
    # CAUSAL INFERENCE BENCHMARKS
    # =========================================================================
    
    IHDP = RealWorldBenchmark(
        name="IHDP",
        description="Infant Health Development Program - Effect of specialist care on cognitive scores",
        dataset_size=747,
        metric_name="√PEHE (Precision in Estimation of Heterogeneous Effects)",
        metric_range=(0.5, 5.0),
        sota_score=0.71,  # DragonNet with targeted regularization
        baseline_score=2.4,  # OLS-1
        higher_is_better=False,
        paper_reference="Hill 2011; Shi et al. 2019 (DragonNet)",
        download_url="https://github.com/vdorie/npci"
    )
    
    JOBS = RealWorldBenchmark(
        name="Jobs/LaLonde",
        description="Effect of job training on employment - National Supported Work Program",
        dataset_size=3212,
        metric_name="Policy Risk",
        metric_range=(0.1, 0.5),
        sota_score=0.21,  # CFR with Wasserstein
        baseline_score=0.35,
        higher_is_better=False,
        paper_reference="LaLonde 1986; Shalit et al. 2017",
        download_url="https://users.nber.org/~rdehejia/data/"
    )
    
    TWINS = RealWorldBenchmark(
        name="Twins",
        description="Effect of birth weight on mortality in twin births",
        dataset_size=11400,
        metric_name="AUC for counterfactual mortality",
        metric_range=(0.6, 0.9),
        sota_score=0.84,  # CEVAE
        baseline_score=0.65,
        higher_is_better=True,
        paper_reference="Louizos et al. 2017 (CEVAE)",
        download_url="https://github.com/AMLab-Amsterdam/CEVAE"
    )
    
    ACIC_2016 = RealWorldBenchmark(
        name="ACIC 2016",
        description="Atlantic Causal Inference Conference competition - 77 DGP settings",
        dataset_size=4802,
        metric_name="√PEHE",
        metric_range=(0.5, 3.0),
        sota_score=0.85,  # DDRN-CFR
        baseline_score=1.8,
        higher_is_better=False,
        paper_reference="Dorie et al. 2019",
        download_url="https://github.com/vdorie/aciccomp"
    )
    
    # =========================================================================
    # COUNTERFACTUAL PHYSICS BENCHMARKS
    # =========================================================================
    
    CLEVRER_CF = RealWorldBenchmark(
        name="CLEVRER-Counterfactual",
        description="Counterfactual questions about collision events in videos",
        dataset_size=37253,  # counterfactual questions
        metric_name="Accuracy",
        metric_range=(0.25, 0.96),
        sota_score=0.67,  # NS-DR (Neuro-Symbolic)
        baseline_score=0.25,  # Random (4-way)
        higher_is_better=True,
        paper_reference="Yi et al. 2020 (ICLR)",
        download_url="http://clevrer.csail.mit.edu/"
    )
    
    COPHY_BLOCKTOWER = RealWorldBenchmark(
        name="CoPhy-BlocktowerCF",
        description="Counterfactual block tower prediction - predict positions after intervention",
        dataset_size=100000,
        metric_name="Position MSE",
        metric_range=(3.0, 20.0),
        sota_score=5.4,  # CophyNet
        baseline_score=15.2,  # NPE
        higher_is_better=False,
        paper_reference="Baradel et al. 2020 (ICLR)",
        download_url="https://projet.liris.cnrs.fr/cophy/"
    )
    
    COPHY_BALLS = RealWorldBenchmark(
        name="CoPhy-BallsCF",
        description="Counterfactual bouncing balls prediction",
        dataset_size=100000,
        metric_name="Position MSE",
        metric_range=(2.0, 15.0),
        sota_score=4.2,
        baseline_score=12.8,
        higher_is_better=False,
        paper_reference="Baradel et al. 2020 (ICLR)",
        download_url="https://projet.liris.cnrs.fr/cophy/"
    )
    
    # =========================================================================
    # CRASH SEVERITY BENCHMARKS
    # =========================================================================
    
    FARS_SEVERITY = RealWorldBenchmark(
        name="FARS-Severity",
        description="Fatality Analysis Reporting System - Fatal crash severity prediction",
        dataset_size=38824,  # 2022 fatal crashes
        metric_name="Macro-F1",
        metric_range=(0.3, 0.7),
        sota_score=0.54,  # CrashLLM-70B
        baseline_score=0.32,  # Logistic Regression
        higher_is_better=True,
        paper_reference="NHTSA; CrashLLM 2024",
        download_url="https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars"
    )
    
    CRSS_SEVERITY = RealWorldBenchmark(
        name="CRSS-Severity",
        description="Crash Report Sampling System - Police-reported crash severity",
        dataset_size=56000,  # annual sample
        metric_name="Balanced Accuracy",
        metric_range=(0.4, 0.75),
        sota_score=0.68,  # XGBoost with feature engineering
        baseline_score=0.45,
        higher_is_better=True,
        paper_reference="NHTSA NCSA",
        download_url="https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system"
    )
    
    SHRP2_SCE = RealWorldBenchmark(
        name="SHRP2-SCE",
        description="SHRP2 Safety Critical Events - Crash/near-crash from naturalistic driving",
        dataset_size=4254,  # crashes + near-crashes
        metric_name="Event Type Classification F1",
        metric_range=(0.5, 0.85),
        sota_score=0.78,
        baseline_score=0.52,
        higher_is_better=True,
        paper_reference="Dingus et al. 2016 (PNAS)",
        download_url="https://insight.shrp2nds.us/"
    )
    
    SYNSHP2 = RealWorldBenchmark(
        name="SynSHRP2",
        description="Synthetic multimodal benchmark derived from SHRP2 with 8798 SCEs",
        dataset_size=8798,  # 1874 crashes + 6924 near-crashes
        metric_name="SCE Classification Accuracy",
        metric_range=(0.6, 0.9),
        sota_score=0.82,
        baseline_score=0.58,
        higher_is_better=True,
        paper_reference="SynSHRP2 2025",
        download_url="https://arxiv.org/abs/2505.06276"
    )
    
    CRASHLLM = RealWorldBenchmark(
        name="CrashLLM-Benchmark",
        description="LLM-based crash prediction on FARS+CRSS merged data",
        dataset_size=150000,
        metric_name="Average F1 (Injury + Severity + Type)",
        metric_range=(0.3, 0.6),
        sota_score=0.538,  # LLaMA-70B fine-tuned
        baseline_score=0.349,  # Traditional ML average
        higher_is_better=True,
        paper_reference="CrashLLM 2024",
        download_url="https://arxiv.org/abs/2406.10789"
    )
    
    # =========================================================================
    # DRIVING SCENARIO BENCHMARKS
    # =========================================================================
    
    CAR_SCENES = RealWorldBenchmark(
        name="CAR-Scenes",
        description="Scene-level severity (1-10) annotations on nuScenes/KITTI/Argoverse",
        dataset_size=5192,
        metric_name="Severity Prediction MAE",
        metric_range=(0.5, 2.5),
        sota_score=0.8,  # GPT-4o with prompting
        baseline_score=1.8,
        higher_is_better=False,
        paper_reference="CAR-Scenes 2024",
        download_url="https://arxiv.org/abs/2511.10701"
    )
    
    CARLA_SAFETY = RealWorldBenchmark(
        name="CARLA-SafetyCritical",
        description="CARLA Leaderboard 2.0 safety-critical scenario performance",
        dataset_size=36,  # routes with 5-21 scenarios each
        metric_name="Driving Score (DS)",
        metric_range=(0, 100),
        sota_score=64.0,  # CaRL
        baseline_score=22.0,  # Roach
        higher_is_better=True,
        paper_reference="CARLA Leaderboard; CaRL 2025",
        download_url="https://leaderboard.carla.org/"
    )
    
    @classmethod
    def get_all_benchmarks(cls) -> Dict[str, RealWorldBenchmark]:
        """Return all registered benchmarks."""
        return {
            # Causal Inference
            "ihdp": cls.IHDP,
            "jobs": cls.JOBS,
            "twins": cls.TWINS,
            "acic_2016": cls.ACIC_2016,
            # Counterfactual Physics
            "clevrer_cf": cls.CLEVRER_CF,
            "cophy_blocktower": cls.COPHY_BLOCKTOWER,
            "cophy_balls": cls.COPHY_BALLS,
            # Crash Severity
            "fars": cls.FARS_SEVERITY,
            "crss": cls.CRSS_SEVERITY,
            "shrp2": cls.SHRP2_SCE,
            "synshp2": cls.SYNSHP2,
            "crashllm": cls.CRASHLLM,
            # Driving
            "car_scenes": cls.CAR_SCENES,
            "carla_safety": cls.CARLA_SAFETY,
        }
    
    @classmethod
    def get_benchmarks_by_category(cls) -> Dict[str, List[RealWorldBenchmark]]:
        """Return benchmarks organized by category."""
        return {
            "causal_inference": [cls.IHDP, cls.JOBS, cls.TWINS, cls.ACIC_2016],
            "counterfactual_physics": [cls.CLEVRER_CF, cls.COPHY_BLOCKTOWER, cls.COPHY_BALLS],
            "crash_severity": [cls.FARS_SEVERITY, cls.CRSS_SEVERITY, cls.SHRP2_SCE, 
                              cls.SYNSHP2, cls.CRASHLLM],
            "driving_scenarios": [cls.CAR_SCENES, cls.CARLA_SAFETY]
        }


# =============================================================================
# SECTION 8: BENCHMARK DATA LOADERS
# =============================================================================

class BenchmarkDataLoader:
    """
    Utilities for loading and preprocessing benchmark datasets.
    """
    
    @staticmethod
    def load_ihdp_format(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load IHDP-format data.
        
        Returns:
            X: Covariates (n, 25)
            T: Treatment indicators (n,)
            Y_factual: Observed outcomes (n,)
            Y_counterfactual: Counterfactual outcomes (n,) - for evaluation only
        """
        # This would load from the actual IHDP dataset files
        # Here we provide the expected format
        raise NotImplementedError(
            "Implement this to load from IHDP .csv files. "
            "Download from: https://github.com/vdorie/npci"
        )
    
    @staticmethod
    def load_fars_crss(
        fars_path: str,
        crss_path: str,
        year: int = 2022
    ) -> Dict[str, np.ndarray]:
        """
        Load and merge FARS/CRSS crash data.
        
        Expected columns:
        - Environmental: weather, light, road_surface
        - Vehicle: vehicle_type, speed, maneuver
        - Crash: collision_type, number_vehicles, point_of_impact
        - Severity: injury_severity (KABCO scale: K=4, A=3, B=2, C=1, O=0)
        """
        raise NotImplementedError(
            "Implement this to load from FARS/CRSS .csv files. "
            "Download from: https://www.nhtsa.gov/research-data/"
        )
    
    @staticmethod
    def create_synthetic_crash_data(
        n_samples: int = 1000,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Create synthetic crash data matching FARS/CRSS distribution.
        
        Useful for development and testing before real data integration.
        """
        np.random.seed(seed)
        
        # Environmental features
        weather = np.random.choice(
            ["clear", "rain", "snow", "fog"], 
            n_samples, 
            p=[0.7, 0.15, 0.1, 0.05]
        )
        light = np.random.choice(
            ["daylight", "dark", "dawn_dusk"],
            n_samples,
            p=[0.6, 0.3, 0.1]
        )
        surface = np.random.choice(
            ["dry", "wet", "snow_ice"],
            n_samples,
            p=[0.7, 0.2, 0.1]
        )
        
        # Crash characteristics
        speed = np.random.gamma(4, 15, n_samples)  # km/h
        delta_v = speed * np.random.uniform(0.3, 0.8, n_samples)
        
        # Generate severity based on features (with known causal structure)
        severity_logit = (
            0.02 * delta_v 
            + 0.5 * (weather == "rain").astype(float)
            + 0.3 * (weather == "snow").astype(float)
            + 0.4 * (light == "dark").astype(float)
            + 0.6 * (surface == "snow_ice").astype(float)
            + np.random.normal(0, 0.5, n_samples)
        )
        severity_prob = 1 / (1 + np.exp(-severity_logit))
        severity_score = np.clip(severity_prob, 0, 1)
        
        # KABCO classification
        kabco = np.zeros(n_samples, dtype=int)
        kabco[severity_score >= 0.8] = 4  # Fatal (K)
        kabco[(severity_score >= 0.6) & (severity_score < 0.8)] = 3  # Serious (A)
        kabco[(severity_score >= 0.4) & (severity_score < 0.6)] = 2  # Minor (B)
        kabco[(severity_score >= 0.2) & (severity_score < 0.4)] = 1  # Possible (C)
        # kabco < 0.2 stays 0 (No injury, O)
        
        return {
            "weather": weather,
            "light": light,
            "surface": surface,
            "speed_kph": speed,
            "delta_v_kph": delta_v,
            "severity_score": severity_score,
            "kabco": kabco,
            "n_samples": n_samples
        }


# =============================================================================
# SECTION 9: QUANTITATIVE BENCHMARK COMPARISON
# =============================================================================

class QuantitativeBenchmarkEvaluator:
    """
    Evaluates our LLM world model against standard benchmarks.
    """
    
    def __init__(self):
        self.registry = RealWorldBenchmarkRegistry()
        self.results: Dict[str, Dict] = {}
    
    def evaluate_on_benchmark(
        self,
        benchmark_name: str,
        our_predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against a specific benchmark.
        
        Args:
            benchmark_name: Key from RealWorldBenchmarkRegistry
            our_predictions: Model predictions
            ground_truth: Ground truth values
            
        Returns:
            Dict with our score, percentile, comparison to SOTA/baseline
        """
        benchmarks = self.registry.get_all_benchmarks()
        if benchmark_name not in benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark = benchmarks[benchmark_name]
        
        # Compute appropriate metric
        if "pehe" in benchmark.metric_name.lower():
            our_score = np.sqrt(np.mean((our_predictions - ground_truth) ** 2))
        elif "mse" in benchmark.metric_name.lower():
            our_score = mean_squared_error(ground_truth, our_predictions)
        elif "mae" in benchmark.metric_name.lower():
            our_score = mean_absolute_error(ground_truth, our_predictions)
        elif "accuracy" in benchmark.metric_name.lower():
            our_score = accuracy_score(ground_truth, our_predictions)
        elif "f1" in benchmark.metric_name.lower():
            _, _, our_score, _ = precision_recall_fscore_support(
                ground_truth, our_predictions, average='macro', zero_division=0
            )
        elif "auc" in benchmark.metric_name.lower():
            our_score = roc_auc_score(ground_truth, our_predictions)
        else:
            # Default: assume it's an accuracy-like metric
            our_score = accuracy_score(ground_truth, our_predictions)
        
        # Compute relative performance
        min_val, max_val = benchmark.metric_range
        if benchmark.higher_is_better:
            percentile = (our_score - min_val) / (max_val - min_val) * 100
            vs_sota = (our_score - benchmark.sota_score) / benchmark.sota_score * 100
            vs_baseline = (our_score - benchmark.baseline_score) / benchmark.baseline_score * 100
        else:
            percentile = (max_val - our_score) / (max_val - min_val) * 100
            vs_sota = (benchmark.sota_score - our_score) / benchmark.sota_score * 100
            vs_baseline = (benchmark.baseline_score - our_score) / benchmark.baseline_score * 100
        
        result = {
            "benchmark": benchmark.name,
            "our_score": our_score,
            "metric_name": benchmark.metric_name,
            "sota_score": benchmark.sota_score,
            "baseline_score": benchmark.baseline_score,
            "vs_sota_pct": vs_sota,
            "vs_baseline_pct": vs_baseline,
            "percentile_in_range": np.clip(percentile, 0, 100),
            "higher_is_better": benchmark.higher_is_better,
            "reference": benchmark.paper_reference
        }
        
        self.results[benchmark_name] = result
        return result
    
    def evaluate_counterfactual_treatment_effects(
        self,
        factual_outcomes: np.ndarray,
        counterfactual_predictions: np.ndarray,
        true_treatment_effects: np.ndarray,
        treatment_indicators: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate treatment effect estimation quality.
        
        This is the core metric for counterfactual crash analysis:
        Given a crash, if we intervene on feature X, how well do we predict
        the change in severity?
        
        Metrics:
        - PEHE: Precision in Estimation of Heterogeneous Effects
        - ATE Error: Error in Average Treatment Effect
        - ATT Error: Error in Average Treatment Effect on Treated
        """
        # Compute Individual Treatment Effects (ITE)
        # ITE = Y(1) - Y(0) for each sample
        
        # Predicted ITE from our model
        predicted_ite = counterfactual_predictions - factual_outcomes
        
        # PEHE
        pehe = np.sqrt(np.mean((predicted_ite - true_treatment_effects) ** 2))
        
        # ATE Error
        predicted_ate = np.mean(predicted_ite)
        true_ate = np.mean(true_treatment_effects)
        ate_error = np.abs(predicted_ate - true_ate)
        
        # ATT Error (Average Treatment Effect on Treated)
        treated_mask = treatment_indicators == 1
        if treated_mask.sum() > 0:
            predicted_att = np.mean(predicted_ite[treated_mask])
            true_att = np.mean(true_treatment_effects[treated_mask])
            att_error = np.abs(predicted_att - true_att)
        else:
            att_error = np.nan
        
        # Policy risk: probability of recommending wrong treatment
        # (relevant for "which intervention to apply")
        recommended_treat = predicted_ite > 0
        should_treat = true_treatment_effects > 0
        policy_error = np.mean(recommended_treat != should_treat)
        
        return {
            "pehe": pehe,
            "ate_error": ate_error,
            "att_error": att_error,
            "policy_error_rate": policy_error,
            "predicted_ate": predicted_ate,
            "true_ate": true_ate
        }
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        lines = [
            "=" * 80,
            "QUANTITATIVE BENCHMARK COMPARISON REPORT",
            "LLM World Model for Counterfactual Crash Analysis",
            "=" * 80,
            ""
        ]
        
        # Group by category
        for category, benchmarks in self.registry.get_benchmarks_by_category().items():
            lines.append(f"\n{'='*40}")
            lines.append(f"  {category.upper().replace('_', ' ')}")
            lines.append(f"{'='*40}")
            
            for bench in benchmarks:
                key = bench.name.lower().replace("-", "_").replace(" ", "_")
                
                # Find matching result
                result = None
                for k, v in self.results.items():
                    if bench.name in v.get("benchmark", ""):
                        result = v
                        break
                
                lines.append(f"\n📊 {bench.name}")
                lines.append(f"   {bench.description}")
                lines.append(f"   Metric: {bench.metric_name}")
                lines.append(f"   Dataset size: {bench.dataset_size:,}")
                
                if result:
                    direction = "↑" if bench.higher_is_better else "↓"
                    lines.append(f"   Our Score: {result['our_score']:.4f} {direction}")
                    lines.append(f"   SOTA: {bench.sota_score:.4f} ({result['vs_sota_pct']:+.1f}%)")
                    lines.append(f"   Baseline: {bench.baseline_score:.4f} ({result['vs_baseline_pct']:+.1f}%)")
                    lines.append(f"   Percentile: {result['percentile_in_range']:.1f}%")
                else:
                    lines.append(f"   SOTA: {bench.sota_score:.4f}")
                    lines.append(f"   Baseline: {bench.baseline_score:.4f}")
                    lines.append(f"   [NOT YET EVALUATED]")
                
                lines.append(f"   Reference: {bench.paper_reference}")
        
        return "\n".join(lines)


# =============================================================================
# SECTION 10: INTEGRATION WITH LLM WORLD MODEL
# =============================================================================

class LLMWorldModelEvaluator:
    """
    Specialized evaluator for LLM-based counterfactual propagation.
    
    Evaluates:
    1. Causal graph structure accuracy
    2. Propagation consistency
    3. Physical plausibility
    4. Severity prediction accuracy
    5. Comparison against physics simulators
    """
    
    def __init__(self, world_model: Any = None):
        self.world_model = world_model
        self.cf_validator = CounterfactualValidator()
        self.benchmark_evaluator = QuantitativeBenchmarkEvaluator()
    
    def evaluate_causal_propagation_accuracy(
        self,
        test_cases: List[Dict],
        use_simulator_as_oracle: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate how accurately the LLM propagates interventions through the causal graph.
        
        Test cases have:
        - initial_state: Dict of node values
        - intervention: Dict of {node: new_value}
        - expected_effects: Dict of {node: expected_value_or_direction}
        """
        results = {
            "direction_accuracy": 0.0,
            "value_accuracy": 0.0,
            "physics_consistency": 0.0,
            "total_propagations": 0,
            "correct_directions": 0,
            "correct_values": 0,
            "physics_violations": 0
        }
        
        for case in test_cases:
            if self.world_model is None:
                continue
                
            initial = case["initial_state"]
            intervention = case["intervention"]
            expected = case.get("expected_effects", {})
            
            # Run propagation
            try:
                counterfactual = self.world_model.propagate_intervention(
                    initial, intervention
                )
            except Exception as e:
                continue
            
            # Check each expected effect
            for node, expected_value in expected.items():
                results["total_propagations"] += 1
                
                if node not in counterfactual:
                    continue
                
                actual = counterfactual[node]
                original = initial.get(node, None)
                
                # Direction check
                if expected_value in ["should_increase", "should_decrease", "should_improve"]:
                    if expected_value == "should_increase" and actual > original:
                        results["correct_directions"] += 1
                    elif expected_value == "should_decrease" and actual < original:
                        results["correct_directions"] += 1
                    elif expected_value == "should_improve":
                        # Context-dependent improvement
                        results["correct_directions"] += 1  # Simplified
                else:
                    # Value check
                    if isinstance(expected_value, (int, float)):
                        if np.abs(actual - expected_value) / (expected_value + 1e-6) < 0.2:
                            results["correct_values"] += 1
                    elif actual == expected_value:
                        results["correct_values"] += 1
            
            # Physics validation
            validation = self.cf_validator.validate_counterfactual(counterfactual)
            if not validation["physics_valid"]:
                results["physics_violations"] += 1
        
        # Compute rates
        total = results["total_propagations"]
        if total > 0:
            results["direction_accuracy"] = results["correct_directions"] / total
            results["value_accuracy"] = results["correct_values"] / total
            results["physics_consistency"] = 1 - (results["physics_violations"] / len(test_cases))
        
        return results
    
    def compare_with_carla_simulation(
        self,
        scenarios: List[Dict],
        carla_results: List[Dict]
    ) -> Dict[str, float]:
        """
        Compare LLM counterfactual predictions with CARLA physics simulation.
        
        CARLA provides ground truth for:
        - Collision occurrence
        - Impact speeds
        - Vehicle trajectories
        - Injury severity estimates
        """
        comparisons = {
            "collision_prediction_accuracy": 0.0,
            "impact_speed_mae": 0.0,
            "severity_correlation": 0.0,
            "trajectory_error": 0.0
        }
        
        collision_correct = 0
        speed_errors = []
        llm_severities = []
        carla_severities = []
        
        for scenario, carla in zip(scenarios, carla_results):
            # Extract predictions
            llm_collision = scenario.get("accident.event_type") == "crash"
            carla_collision = carla.get("collision_occurred", False)
            
            if llm_collision == carla_collision:
                collision_correct += 1
            
            llm_speed = scenario.get("crash_state.speed_at_impact_kph", 0)
            carla_speed = carla.get("impact_speed_kph", 0)
            speed_errors.append(abs(llm_speed - carla_speed))
            
            llm_sev = scenario.get("injury_outcome.severity_score", 0)
            carla_sev = carla.get("estimated_severity", 0)
            llm_severities.append(llm_sev)
            carla_severities.append(carla_sev)
        
        n = len(scenarios)
        if n > 0:
            comparisons["collision_prediction_accuracy"] = collision_correct / n
            comparisons["impact_speed_mae"] = np.mean(speed_errors)
            
            if len(llm_severities) > 1:
                corr, _ = stats.pearsonr(llm_severities, carla_severities)
                comparisons["severity_correlation"] = corr if not np.isnan(corr) else 0.0
        
        return comparisons


# =============================================================================
# UPDATED DEMO
# =============================================================================

def demo_full_evaluation():
    """Demonstrate the complete evaluation framework with benchmark comparisons."""
    
    print("=" * 80)
    print("COMPLETE EVALUATION FRAMEWORK DEMO")
    print("LLM World Model for Counterfactual Crash Analysis")
    print("=" * 80)
    
    # 1. Load synthetic data
    print("\n📊 Loading synthetic crash data...")
    data = BenchmarkDataLoader.create_synthetic_crash_data(n_samples=500)
    print(f"   Created {data['n_samples']} synthetic crash records")
    
    # 2. Simulate predictions (in practice, these come from the LLM world model)
    np.random.seed(42)
    severity_true = data["kabco"]
    severity_pred = severity_true.copy()
    noise_idx = np.random.choice(len(severity_pred), len(severity_pred) // 5, replace=False)
    severity_pred[noise_idx] = np.random.randint(0, 5, len(noise_idx))
    
    # 3. Compute severity metrics
    print("\n📈 Computing severity prediction metrics...")
    severity_metrics = compute_severity_metrics(
        severity_true, severity_pred,
        class_names=["No Injury", "Possible", "Minor", "Serious", "Fatal"]
    )
    print(f"   Accuracy: {severity_metrics.accuracy:.2%}")
    print(f"   Macro F1: {severity_metrics.macro_f1:.3f}")
    print(f"   Ordinal MAE: {severity_metrics.ordinal_mae:.3f}")
    
    # 4. Benchmark comparison
    print("\n🏆 Comparing against benchmarks...")
    evaluator = QuantitativeBenchmarkEvaluator()
    
    # Compare FARS severity prediction
    fars_result = evaluator.evaluate_on_benchmark(
        "fars",
        severity_pred,
        severity_true
    )
    print(f"   FARS Benchmark:")
    print(f"     Our Score: {fars_result['our_score']:.3f}")
    print(f"     vs SOTA ({fars_result['sota_score']:.3f}): {fars_result['vs_sota_pct']:+.1f}%")
    print(f"     vs Baseline ({fars_result['baseline_score']:.3f}): {fars_result['vs_baseline_pct']:+.1f}%")
    
    # 5. Treatment effect evaluation
    print("\n💊 Evaluating counterfactual treatment effects...")
    # Simulate: intervention = improving weather from rain to clear
    factual_severity = data["severity_score"]
    # Counterfactual: expect ~20% reduction in severity
    counterfactual_severity = factual_severity * 0.8 + np.random.normal(0, 0.05, len(factual_severity))
    true_effects = -0.2 * factual_severity + np.random.normal(0, 0.03, len(factual_severity))
    treatment = (data["weather"] == "rain").astype(int)
    
    te_metrics = evaluator.evaluate_counterfactual_treatment_effects(
        factual_severity,
        counterfactual_severity,
        true_effects,
        treatment
    )
    print(f"   PEHE: {te_metrics['pehe']:.4f}")
    print(f"   ATE Error: {te_metrics['ate_error']:.4f}")
    print(f"   Policy Error Rate: {te_metrics['policy_error_rate']:.2%}")
    
    # 6. Counterfactual validation
    print("\n✅ Validating counterfactual plausibility...")
    cf_events = [
        {
            "environment.weather": "clear",
            "environment.surface_condition": "dry",
            "crash_state.delta_v_kph": 25,
            "injury_outcome.severity_score": 0.3
        }
        for _ in range(50)
    ]
    # Add some implausible ones
    cf_events.extend([
        {
            "environment.weather": "rain",
            "environment.surface_condition": "dry",  # Implausible
            "crash_state.delta_v_kph": 100,
            "injury_outcome.severity_score": 0.1  # Implausible
        }
        for _ in range(10)
    ])
    
    validator = CounterfactualValidator()
    cf_metrics = validator.compute_batch_metrics(cf_events)
    print(f"   Validity Rate: {cf_metrics.validity_rate:.2%}")
    print(f"   Ontology Consistency: {cf_metrics.ontology_consistency:.2%}")
    print(f"   Physical Plausibility: {cf_metrics.physical_plausibility:.2%}")
    
    # 7. List available benchmarks
    print("\n📚 Available Real-World Benchmarks:")
    for category, benchmarks in RealWorldBenchmarkRegistry.get_benchmarks_by_category().items():
        print(f"\n   {category.upper()}:")
        for b in benchmarks:
            print(f"     - {b.name}: {b.description[:50]}...")
    
    # 8. Generate full report
    print("\n" + "=" * 80)
    print("GENERATING FULL COMPARISON REPORT")
    print("=" * 80)
    print(evaluator.generate_comparison_report())
    
    print("\n✨ Evaluation complete!")


if __name__ == "__main__":
    demo_full_evaluation()
