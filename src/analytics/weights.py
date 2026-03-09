"""Weight evolution analytics for AdCraft.

After sufficient ads have been evaluated, calculates Pearson correlations
between each dimension's scores and the weighted average, then compares
observed correlations to the initial weight assignments. Advisory only —
recommendations are logged but not auto-applied.
"""

from __future__ import annotations

import math
import sqlite3

from src.decisions.logger import log_decision
from src.evaluate.rubrics import DIMENSION_WEIGHTS, DIMENSIONS


class WeightEvolver:
    """Analyzes whether initial dimension weights match observed quality signals."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        min_sample_size: int = 50,
    ) -> None:
        self._conn = conn
        self._min_sample_size = min_sample_size
        self._initial_weights = dict(DIMENSION_WEIGHTS)

        log_decision(
            "analytics",
            "weight_evolver_init",
            f"WeightEvolver initialized: min_sample_size={min_sample_size}, "
            f"initial_weights={self._initial_weights}",
            {
                "min_sample_size": min_sample_size,
                "initial_weights": self._initial_weights,
            },
        )

    def calculate_correlations(self) -> dict[str, float]:
        """Compute Pearson correlation between each dimension and weighted average.

        Queries all evaluation scores from DB, groups by ad_id, computes
        weighted averages, then correlates each dimension against that average.
        """
        # Fetch all evaluation rows
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT ad_id, dimension, score FROM evaluations ORDER BY ad_id"
        ).fetchall()

        if not rows:
            log_decision(
                "analytics",
                "no_evaluation_data",
                "No evaluation data found in database",
                {},
            )
            return {}

        # Group scores by ad_id
        ad_scores: dict[str, dict[str, float]] = {}
        for row in rows:
            ad_id = row["ad_id"]
            if ad_id not in ad_scores:
                ad_scores[ad_id] = {}
            ad_scores[ad_id][row["dimension"]] = float(row["score"])

        # Filter to ads with all dimensions scored
        complete_ads: dict[str, dict[str, float]] = {
            ad_id: scores
            for ad_id, scores in ad_scores.items()
            if all(d in scores for d in DIMENSIONS)
        }

        if not complete_ads:
            log_decision(
                "analytics",
                "no_complete_evaluations",
                "No ads with complete dimension scores found",
                {"total_ads": len(ad_scores), "required_dimensions": DIMENSIONS},
            )
            return {}

        # Compute weighted averages per ad
        weighted_avgs: list[float] = []
        dim_score_lists: dict[str, list[float]] = {d: [] for d in DIMENSIONS}

        for scores in complete_ads.values():
            w_avg = sum(scores[d] * self._initial_weights[d] for d in DIMENSIONS)
            weighted_avgs.append(w_avg)
            for d in DIMENSIONS:
                dim_score_lists[d].append(scores[d])

        # Calculate Pearson correlation for each dimension vs weighted average
        correlations: dict[str, float] = {}
        for dim in DIMENSIONS:
            r = _pearson(dim_score_lists[dim], weighted_avgs)
            correlations[dim] = r

        log_decision(
            "analytics",
            "correlations_calculated",
            f"Pearson correlations computed for {len(complete_ads)} ads: "
            + ", ".join(f"{d}={r:.3f}" for d, r in correlations.items()),
            {
                "sample_size": len(complete_ads),
                "correlations": correlations,
            },
        )

        return correlations

    def compare_to_initial_weights(
        self, correlations: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        """Flag dimensions where weight and correlation diverge significantly.

        Returns a dict per dimension with weight, correlation, and divergence.
        """
        if not correlations:
            return {}

        comparison: dict[str, dict[str, float]] = {}
        for dim in DIMENSIONS:
            weight = self._initial_weights.get(dim, 0.0)
            corr = correlations.get(dim, 0.0)
            divergence = abs(weight - corr)
            comparison[dim] = {
                "weight": weight,
                "correlation": corr,
                "divergence": divergence,
            }

        flagged = {d: v for d, v in comparison.items() if v["divergence"] > 0.15}

        if flagged:
            log_decision(
                "analytics",
                "weight_divergence_flagged",
                f"{len(flagged)} dimensions show significant weight-correlation divergence: "
                + ", ".join(
                    f"{d} (w={v['weight']:.2f}, r={v['correlation']:.3f})"
                    for d, v in flagged.items()
                ),
                {"flagged": flagged, "all": comparison},
            )
        else:
            log_decision(
                "analytics",
                "weights_validated",
                "All dimension weights align with observed correlations (divergence < 0.15)",
                {"comparison": comparison},
            )

        return comparison

    def recommend_weights(self, correlations: dict[str, float]) -> dict[str, float]:
        """Normalize correlations to sum to 1.0 as recommended weights.

        Clamps negative correlations to a small positive floor (0.05) so
        every dimension retains some weight.
        """
        if not correlations:
            return dict(self._initial_weights)

        # Floor negative correlations
        clamped = {d: max(c, 0.05) for d, c in correlations.items()}
        total = sum(clamped.values())

        if total == 0:
            return dict(self._initial_weights)

        recommended = {d: round(v / total, 4) for d, v in clamped.items()}

        log_decision(
            "analytics",
            "weights_recommended",
            "Recommended weights (normalized correlations): "
            + ", ".join(f"{d}={w:.3f}" for d, w in recommended.items()),
            {"recommended": recommended, "current": self._initial_weights},
        )

        return recommended

    def evolve(self) -> dict:
        """Run the full weight evolution analysis if sample size is met.

        Returns analysis results dict with correlations, comparison, and
        recommendations.
        """
        # Check sample size
        row = self._conn.execute("SELECT COUNT(DISTINCT ad_id) as cnt FROM evaluations").fetchone()
        sample_count = row[0] if row else 0

        if sample_count < self._min_sample_size:
            log_decision(
                "analytics",
                "insufficient_sample",
                f"Only {sample_count}/{self._min_sample_size} ads evaluated. "
                f"Skipping weight evolution.",
                {
                    "sample_count": sample_count,
                    "min_required": self._min_sample_size,
                },
            )
            return {
                "status": "insufficient_data",
                "sample_count": sample_count,
                "min_required": self._min_sample_size,
            }

        log_decision(
            "analytics",
            "evolve_start",
            f"Running weight evolution with {sample_count} ads "
            f"(threshold: {self._min_sample_size})",
            {"sample_count": sample_count},
        )

        correlations = self.calculate_correlations()
        comparison = self.compare_to_initial_weights(correlations)
        recommended = self.recommend_weights(correlations)

        result = {
            "status": "complete",
            "sample_count": sample_count,
            "correlations": correlations,
            "comparison": comparison,
            "recommended_weights": recommended,
            "current_weights": dict(self._initial_weights),
        }

        log_decision(
            "analytics",
            "evolve_complete",
            f"Weight evolution complete for {sample_count} ads. "
            f"Recommendation: " + ", ".join(f"{d}={w:.3f}" for d, w in recommended.items()),
            result,
        )

        return result


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists.

    Returns 0.0 if variance is zero or lists are too short.
    Uses stdlib math only — no numpy/scipy.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0 or den_y == 0:
        return 0.0

    return round(num / (den_x * den_y), 6)
