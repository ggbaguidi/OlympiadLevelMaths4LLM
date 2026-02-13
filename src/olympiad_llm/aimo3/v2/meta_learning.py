# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Meta-learning components for adaptive strategy selection.

Implements contextual bandits with k-NN warm start for problem-specific
strategy optimization. Uses TRIZ principles:
- Principle 17: Another Dimension (add problem similarity space)
- Principle 23: Feedback (learn from attempt outcomes)
- Principle 40: Composite (combine exploration + exploitation)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProblemFeatures:
    """Lightweight feature vector for problem characterization."""

    # Domain indicators (0-1 scores)
    has_modular_arithmetic: float = 0.0
    has_combinatorics: float = 0.0
    has_number_theory: float = 0.0
    has_algebra: float = 0.0
    has_geometry: float = 0.0

    # Complexity indicators
    word_count: int = 0
    numeric_count: int = 0
    equation_count: int = 0
    max_number_magnitude: float = 0.0

    # Structural features
    has_proof_requirement: float = 0.0
    has_constraint_satisfaction: float = 0.0
    has_optimization: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for similarity computation."""
        return np.array(
            [
                self.has_modular_arithmetic,
                self.has_combinatorics,
                self.has_number_theory,
                self.has_algebra,
                self.has_geometry,
                min(self.word_count / 500, 1.0),  # Normalize
                min(self.numeric_count / 50, 1.0),
                min(self.equation_count / 10, 1.0),
                min(self.max_number_magnitude / 10, 1.0),
                self.has_proof_requirement,
                self.has_constraint_satisfaction,
                self.has_optimization,
            ],
            dtype=np.float32,
        )

    def similarity(self, other: ProblemFeatures) -> float:
        """Cosine similarity between two problem feature vectors."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)


class ProblemEmbedder:
    """Lightweight problem embedding without heavy ML dependencies."""

    # Keywords for domain detection
    MODULAR_KEYWORDS = [
        "mod",
        "modulo",
        "remainder",
        "divisible",
        "congruence",
        "residue",
        "modular",
        "≡",
    ]
    COMBINATORICS_KEYWORDS = [
        "combination",
        "permutation",
        "choose",
        "binomial",
        "subset",
        "arrangement",
        "count",
        "ways",
        "pairs",
        "triples",
    ]
    NUMBER_THEORY_KEYWORDS = [
        "prime",
        "divisor",
        "factor",
        "gcd",
        "lcm",
        "totient",
        "coprime",
        "Euler",
        "Fermat",
    ]
    ALGEBRA_KEYWORDS = [
        "equation",
        "solve",
        "polynomial",
        "root",
        "factor",
        "expand",
        "simplify",
    ]
    GEOMETRY_KEYWORDS = [
        "triangle",
        "circle",
        "angle",
        "side",
        "area",
        "volume",
        "point",
        "line",
        "plane",
        "coordinate",
    ]

    def embed(self, problem_text: str) -> ProblemFeatures:
        """Extract features from problem text."""
        text_lower = problem_text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        numbers = re.findall(r"\b\d+\b", problem_text)

        features = ProblemFeatures()

        # Word and numeric counts
        features.word_count = len(words)
        features.numeric_count = len(numbers)

        # Max number magnitude
        if numbers:
            max_num = max(int(n) for n in numbers)
            features.max_number_magnitude = math.log10(max_num) if max_num > 0 else 0

        # Domain detection via keyword frequency
        text_words = set(words)
        features.has_modular_arithmetic = self._keyword_score(
            text_words, self.MODULAR_KEYWORDS
        )
        features.has_combinatorics = self._keyword_score(
            text_words, self.COMBINATORICS_KEYWORDS
        )
        features.has_number_theory = self._keyword_score(
            text_words, self.NUMBER_THEORY_KEYWORDS
        )
        features.has_algebra = self._keyword_score(text_words, self.ALGEBRA_KEYWORDS)
        features.has_geometry = self._keyword_score(text_words, self.GEOMETRY_KEYWORDS)

        # Equation detection
        features.equation_count = len(
            re.findall(r"[=<>]|\\leq|\\geq|\\neq", problem_text)
        )

        # Structural features
        features.has_proof_requirement = float(
            any(
                kw in text_lower
                for kw in ["prove", "show that", "demonstrate", "verify"]
            )
        )
        features.has_constraint_satisfaction = float(
            features.equation_count >= 2 or "subject to" in text_lower
        )
        features.has_optimization = float(
            any(
                kw in text_lower
                for kw in [
                    "maximum",
                    "minimum",
                    "maximize",
                    "minimize",
                    "largest",
                    "smallest",
                ]
            )
        )

        return features

    @staticmethod
    def _keyword_score(text_words: set, keywords: List[str]) -> float:
        """Calculate normalized keyword presence score."""
        matches = sum(1 for kw in keywords if kw in text_words)
        return min(matches / 3, 1.0)  # Cap at 3+ matches


@dataclass
class StrategyExperience:
    """Record of strategy performance on a specific problem type."""

    strategy_name: str
    problem_features: ProblemFeatures
    success: bool
    attempts_to_success: int
    time_spent: float
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class StrategyBandit:
    """Thompson Sampling contextual bandit for strategy selection.

    Learns which strategies work best for which problem types using
    Bayesian updating. Addresses TRIZ Principle 23 (Feedback).
    """

    def __init__(
        self,
        strategy_names: List[str],
        exploration_factor: float = 1.0,
        similarity_threshold: float = 0.7,
        experience_file: Optional[Path] = None,
    ):
        self.strategy_names = strategy_names
        self.exploration_factor = exploration_factor
        self.similarity_threshold = similarity_threshold
        self.experience_file = experience_file

        # Beta distribution parameters: Beta(alpha, beta)
        # alpha = successes + 1, beta = failures + 1
        # Key: (strategy_name, problem_cluster_id)
        self.alpha: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
        self.beta: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)

        # Raw experience buffer for k-NN
        self.experiences: List[StrategyExperience] = []

        # Problem cluster assignments
        self.cluster_centers: List[ProblemFeatures] = []
        self.next_cluster_id = 0

        if experience_file and experience_file.exists():
            self._load_experiences()

    def _get_or_create_cluster(self, features: ProblemFeatures) -> str:
        """Assign problem to existing cluster or create new one."""
        if not self.cluster_centers:
            cluster_id = f"cluster_{self.next_cluster_id}"
            self.cluster_centers.append(features)
            self.next_cluster_id += 1
            return cluster_id

        # Find most similar cluster
        similarities = [
            (i, features.similarity(center))
            for i, center in enumerate(self.cluster_centers)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        best_idx, best_sim = similarities[0]

        if best_sim >= self.similarity_threshold:
            return f"cluster_{best_idx}"
        else:
            # Create new cluster
            cluster_id = f"cluster_{self.next_cluster_id}"
            self.cluster_centers.append(features)
            self.next_cluster_id += 1
            return cluster_id

    def select_strategy(
        self,
        problem_features: ProblemFeatures,
        attempt_idx: int,
        used_strategies: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Select strategy using Thompson Sampling with k-NN warm start.

        Returns:
            Tuple of (strategy_name, metadata_dict)
        """
        used_strategies = used_strategies or []
        cluster_id = self._get_or_create_cluster(problem_features)

        # For first attempts, use k-NN to warm start
        if attempt_idx == 0 and self.experiences:
            strategy_scores = self._knn_score_strategies(
                problem_features, used_strategies
            )
            if strategy_scores:
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                return best_strategy, {
                    "method": "knn_warm_start",
                    "cluster": cluster_id,
                    "knn_scores": strategy_scores,
                    "exploration": False,
                }

        # Thompson Sampling: sample from Beta distributions
        samples = {}
        for strategy in self.strategy_names:
            if strategy in used_strategies:
                continue  # Don't repeat strategies

            alpha = self.alpha[(strategy, cluster_id)]
            beta = self.beta[(strategy, cluster_id)]

            # Thompson sample
            sample = np.random.beta(alpha, beta)

            # Add exploration bonus for under-sampled strategies
            total_trials = alpha + beta - 2  # Subtract prior
            if total_trials < 5:
                sample += self.exploration_factor * (5 - total_trials) * 0.1

            samples[strategy] = sample

        if not samples:
            # All strategies used, fallback to rotation
            fallback = self.strategy_names[attempt_idx % len(self.strategy_names)]
            return fallback, {"method": "fallback_rotation", "cluster": cluster_id}

        best_strategy = max(samples, key=samples.get)
        return best_strategy, {
            "method": "thompson_sampling",
            "cluster": cluster_id,
            "samples": samples,
            "exploration": self.alpha[(best_strategy, cluster_id)]
            + self.beta[(best_strategy, cluster_id)]
            < 10,  # Still exploring
        }

    def _knn_score_strategies(
        self, features: ProblemFeatures, exclude: List[str]
    ) -> Dict[str, float]:
        """Score strategies based on k-NN similarity."""
        k = min(5, len(self.experiences))
        if k == 0:
            return {}

        # Find k most similar past experiences
        scored = [
            (exp, features.similarity(exp.problem_features))
            for exp in self.experiences
            if exp.strategy_name not in exclude
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        # Weighted score by similarity
        strategy_scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for exp, sim in top_k:
            if sim > 0.3:  # Minimum similarity threshold
                reward = 1.0 / max(exp.attempts_to_success, 1) if exp.success else 0.0
                strategy_scores[exp.strategy_name].append((reward, sim))

        # Aggregate scores
        final_scores = {}
        for strategy, rewards_sims in strategy_scores.items():
            total_weight = sum(sim for _, sim in rewards_sims)
            if total_weight > 0:
                weighted_score = sum(r * s for r, s in rewards_sims) / total_weight
                final_scores[strategy] = weighted_score

        return final_scores

    def update(
        self,
        problem_features: ProblemFeatures,
        strategy_name: str,
        success: bool,
        attempts_to_success: int = 1,
        time_spent: float = 0.0,
    ) -> None:
        """Update bandit with outcome (Bayesian update)."""
        cluster_id = self._get_or_create_cluster(problem_features)

        # Record experience
        exp = StrategyExperience(
            strategy_name=strategy_name,
            problem_features=problem_features,
            success=success,
            attempts_to_success=attempts_to_success,
            time_spent=time_spent,
        )
        self.experiences.append(exp)

        # Bayesian update
        key = (strategy_name, cluster_id)
        if success:
            self.alpha[key] += 1.0 / attempts_to_success  # Weight by efficiency
        else:
            self.beta[key] += 1.0

        # Save to disk periodically
        if len(self.experiences) % 10 == 0 and self.experience_file:
            self._save_experiences()

    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies across clusters."""
        stats = {}
        for strategy in self.strategy_names:
            total_successes = sum(
                v for (s, _), v in self.alpha.items() if s == strategy
            ) - len([c for s, c in self.alpha.keys() if s == strategy])
            total_failures = sum(
                v for (s, _), v in self.beta.items() if s == strategy
            ) - len([c for s, c in self.beta.keys() if s == strategy])
            total = total_successes + total_failures

            stats[strategy] = {
                "successes": total_successes,
                "failures": total_failures,
                "total_trials": total,
                "success_rate": total_successes / total if total > 0 else 0.5,
            }
        return stats

    def _save_experiences(self) -> None:
        """Persist experiences to disk."""
        if not self.experience_file:
            return
        try:
            import pickle

            data = {
                "alpha": dict(self.alpha),
                "beta": dict(self.beta),
                "experiences": self.experiences,
                "cluster_centers": self.cluster_centers,
                "next_cluster_id": self.next_cluster_id,
            }
            self.experience_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.experience_file, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Best-effort persistence

    def _load_experiences(self) -> None:
        """Load experiences from disk."""
        if not self.experience_file or not self.experience_file.exists():
            return
        try:
            import pickle

            with open(self.experience_file, "rb") as f:
                data = pickle.load(f)

            self.alpha = defaultdict(lambda: 1.0, data.get("alpha", {}))
            self.beta = defaultdict(lambda: 1.0, data.get("beta", {}))
            self.experiences = data.get("experiences", [])
            self.cluster_centers = data.get("cluster_centers", [])
            self.next_cluster_id = data.get("next_cluster_id", 0)
        except Exception:
            pass  # Start fresh if corrupted


class AdaptiveHyperparameters:
    """Meta-learn optimal hyperparameters per problem type."""

    def __init__(self, default_config: Any):
        self.default_config = default_config
        self.problem_type_configs: Dict[str, Dict[str, Any]] = {}

    def get_config(self, problem_features: ProblemFeatures) -> Dict[str, Any]:
        """Get hyperparameters adapted to problem type."""
        # Determine problem type
        features = problem_features.to_vector()
        domain_idx = np.argmax(features[:5])  # First 5 are domain indicators
        domains = ["modular", "combinatorics", "number_theory", "algebra", "geometry"]
        problem_type = domains[domain_idx] if domain_idx < 5 else "general"

        if problem_type not in self.problem_type_configs:
            return self._default_for_type(problem_type)

        return self.problem_type_configs[problem_type]

    def _default_for_type(self, problem_type: str) -> Dict[str, Any]:
        """Get sensible defaults based on problem type."""
        defaults = {
            "modular": {
                "attempts": 8,
                "temperature": 0.8,
                "early_stop": 3,
                "preferred_strategy": "modular_arithmetic",
            },
            "combinatorics": {
                "attempts": 10,
                "temperature": 0.9,
                "early_stop": 4,
                "preferred_strategy": "generate_and_test",
            },
            "number_theory": {
                "attempts": 8,
                "temperature": 0.85,
                "early_stop": 3,
                "preferred_strategy": "reduce_to_known",
            },
            "algebra": {
                "attempts": 6,
                "temperature": 0.7,
                "early_stop": 2,
                "preferred_strategy": "algebraic_manipulation",
            },
            "geometry": {
                "attempts": 8,
                "temperature": 0.8,
                "early_stop": 3,
                "preferred_strategy": "work_backwards",
            },
            "general": {
                "attempts": 8,
                "temperature": 0.95,
                "early_stop": 3,
                "preferred_strategy": None,
            },
        }
        return defaults.get(problem_type, defaults["general"])

    def update_from_outcome(
        self,
        problem_features: ProblemFeatures,
        config_used: Dict[str, Any],
        success: bool,
        time_spent: float,
    ) -> None:
        """Update hyperparameters based on outcome (gradient-free optimization)."""
        features = problem_features.to_vector()
        domain_idx = np.argmax(features[:5])
        domains = ["modular", "combinatorics", "number_theory", "algebra", "geometry"]
        problem_type = domains[domain_idx] if domain_idx < 5 else "general"

        # Simple heuristic: if succeeded quickly, reinforce config
        if success and time_spent < 120:
            self.problem_type_configs[problem_type] = config_used


# Global singleton for stateful learning across solver instances
_BANDIT_INSTANCE: Optional[StrategyBandit] = None
_EMBEDDER_INSTANCE: Optional[ProblemEmbedder] = None


def get_global_bandit(
    strategy_names: Optional[List[str]] = None,
    experience_file: Optional[Path] = None,
) -> StrategyBandit:
    """Get or create global bandit instance."""
    global _BANDIT_INSTANCE
    if _BANDIT_INSTANCE is None:
        if strategy_names is None:
            raise ValueError("strategy_names required for bandit initialization")
        _BANDIT_INSTANCE = StrategyBandit(
            strategy_names=strategy_names,
            experience_file=experience_file,
        )
    return _BANDIT_INSTANCE


def get_global_embedder() -> ProblemEmbedder:
    """Get or create global embedder instance."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = ProblemEmbedder()
    return _EMBEDDER_INSTANCE


def reset_global_state() -> None:
    """Reset global meta-learning state (for testing)."""
    global _BANDIT_INSTANCE, _EMBEDDER_INSTANCE
    _BANDIT_INSTANCE = None
    _EMBEDDER_INSTANCE = None
