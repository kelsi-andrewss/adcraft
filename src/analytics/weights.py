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

    def _fetch_complete_ads(self) -> dict[str, dict[str, float]]:
        """Query evaluation rows, build complete_ads dict (ads with all DIMENSIONS)."""
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT ad_id, dimension, score FROM evaluations ORDER BY ad_id"
        ).fetchall()

        if not rows:
            return {}

        ad_scores: dict[str, dict[str, float]] = {}
        for row in rows:
            ad_id = row["ad_id"]
            if ad_id not in ad_scores:
                ad_scores[ad_id] = {}
            ad_scores[ad_id][row["dimension"]] = float(row["score"])

        return {
            ad_id: scores
            for ad_id, scores in ad_scores.items()
            if all(d in scores for d in DIMENSIONS)
        }

    def _compute_weighted_scores(
        self,
        complete_ads: dict[str, dict[str, float]],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute weighted average score per ad."""
        return {
            ad_id: sum(scores[d] * weights[d] for d in DIMENSIONS)
            for ad_id, scores in complete_ads.items()
        }

    def calculate_correlations(self) -> dict[str, float]:
        """Compute Pearson correlation between each dimension and weighted average.

        Queries all evaluation scores from DB, groups by ad_id, computes
        weighted averages, then correlates each dimension against that average.
        """
        complete_ads = self._fetch_complete_ads()

        if not complete_ads:
            log_decision(
                "analytics",
                "no_complete_evaluations",
                "No ads with complete dimension scores found",
                {},
            )
            return {}

        weighted_scores = self._compute_weighted_scores(complete_ads, self._initial_weights)
        weighted_avgs = list(weighted_scores.values())

        dim_score_lists: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        for ad_id in weighted_scores:
            for d in DIMENSIONS:
                dim_score_lists[d].append(complete_ads[ad_id][d])

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

    def run_shadow_analysis(self, recommended_weights: dict[str, float]) -> dict:
        """Score all complete ads under current and recommended weights, compare.

        Returns comparison metrics including MAD, Pearson correlation,
        top-decile ranking shift, and score deltas summary.
        """
        complete_ads = self._fetch_complete_ads()
        sample_count = len(complete_ads)

        if sample_count < self._min_sample_size:
            log_decision(
                "analytics",
                "shadow_insufficient_data",
                f"Shadow analysis skipped: only {sample_count} complete ads "
                f"(need {self._min_sample_size})",
                {"sample_count": sample_count},
            )
            return {"status": "insufficient_data", "sample_count": sample_count}

        current_scores = self._compute_weighted_scores(complete_ads, self._initial_weights)
        recommended_scores = self._compute_weighted_scores(complete_ads, recommended_weights)

        ad_ids = list(current_scores.keys())
        current_vals = [current_scores[a] for a in ad_ids]
        recommended_vals = [recommended_scores[a] for a in ad_ids]

        # Mean absolute deviation
        deltas = [recommended_vals[i] - current_vals[i] for i in range(sample_count)]
        abs_deltas = [abs(d) for d in deltas]
        mean_absolute_deviation = round(sum(abs_deltas) / sample_count, 6)

        # Pearson correlation between the two score sets
        pearson_correlation = _pearson(current_vals, recommended_vals)

        # Score deltas summary
        mean_delta = round(sum(deltas) / sample_count, 6)
        max_positive_delta = round(max(deltas), 6)
        max_negative_delta = round(min(deltas), 6)

        # Top-decile ranking shift
        top_k = math.ceil(sample_count * 0.10)
        current_ranked = sorted(ad_ids, key=lambda a: current_scores[a], reverse=True)
        recommended_ranked = sorted(ad_ids, key=lambda a: recommended_scores[a], reverse=True)
        current_top = set(current_ranked[:top_k])
        recommended_top = set(recommended_ranked[:top_k])
        top_decile_shift_count = len(current_top - recommended_top)
        top_decile_shift_pct = round(top_decile_shift_count / top_k, 4) if top_k > 0 else 0.0

        result = {
            "status": "complete",
            "sample_count": sample_count,
            "mean_absolute_deviation": mean_absolute_deviation,
            "pearson_correlation": pearson_correlation,
            "top_decile_shift_count": top_decile_shift_count,
            "top_decile_shift_pct": top_decile_shift_pct,
            "current_weights": dict(self._initial_weights),
            "recommended_weights": dict(recommended_weights),
            "score_deltas_summary": {
                "mean_delta": mean_delta,
                "max_positive_delta": max_positive_delta,
                "max_negative_delta": max_negative_delta,
            },
        }

        log_decision(
            "analytics",
            "shadow_analysis_complete",
            f"Shadow analysis on {sample_count} ads: "
            f"MAD={mean_absolute_deviation:.4f}, "
            f"r={pearson_correlation:.4f}, "
            f"top-decile shift={top_decile_shift_count}/{top_k} "
            f"({top_decile_shift_pct:.1%})",
            result,
        )

        return result

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

        if recommended != self._initial_weights:
            result["shadow_analysis"] = self.run_shadow_analysis(recommended)

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
