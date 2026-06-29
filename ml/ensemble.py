"""
ensemble.py
============
Combines the CNN classifier's calibrated probabilities with the classical
BLS/vetting features into a single, explainable final verdict. This mirrors
how real Kepler/TESS vetting works: an automated classifier proposes a
disposition, but domain-expert rules (odd-even test, secondary eclipse test,
V-shape test, centroid/blend checks) can veto or downweight it.

Output: a CandidateVerdict with:
  - final_label, final_confidence (calibrated + rule-adjusted)
  - per-class probability breakdown
  - a list of "evidence" reasons (positive and negative) driving the verdict
  - a False-Positive risk flag with the specific triggering test(s)

This is the layer that makes the system's predictions explainable rather
than a black-box softmax number.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from features import TransitFeatures
from model_cnn import CLASS_NAMES


@dataclass
class EvidenceItem:
    direction: str       # "supports" | "against"
    label: str           # which class this evidence relates to
    reason: str           # human-readable explanation
    strength: float       # 0-1, how strong this piece of evidence is


@dataclass
class CandidateVerdict:
    final_label: str
    final_confidence: float
    cnn_probabilities: dict           # {class_name: prob} from calibrated CNN
    rule_adjusted_probabilities: dict  # after ensemble fusion with classical rules
    evidence: list                     # list[EvidenceItem]
    false_positive_flags: list         # list[str] - specific FP test names that triggered
    is_likely_false_positive: bool
    data_quality_warning: str | None   # e.g. "only 2 transits observed - period uncertain"

    def to_dict(self):
        d = {
            "final_label": self.final_label,
            "final_confidence": self.final_confidence,
            "cnn_probabilities": self.cnn_probabilities,
            "rule_adjusted_probabilities": self.rule_adjusted_probabilities,
            "evidence": [vars(e) for e in self.evidence],
            "false_positive_flags": self.false_positive_flags,
            "is_likely_false_positive": self.is_likely_false_positive,
            "data_quality_warning": self.data_quality_warning,
        }
        return d


class EnsembleScorer:
    """
    Rule thresholds below are calibrated against the classical Kepler/TESS
    vetting literature (Coughlin et al. 2016 KOI FP catalog; Thompson et al.
    2018 DR25 robovetter) at a level appropriate for a hackathon-scale system
    — i.e. directionally correct and explainable, not a full robovetter replica.
    """

    ODD_EVEN_FP_SIGMA = 3.0          # >3 sigma odd/even depth mismatch -> likely EB
    SECONDARY_ECLIPSE_FP_SIGMA = 5.5  # >5.5 sigma secondary eclipse -> likely EB
    SHAPE_SCORE_FP_THRESHOLD = 0.25   # very V-shaped/grazing -> EB-like
    MIN_SNR_RELIABLE = 7.0            # below this, treat as marginal/noise-dominated
    MIN_TRANSITS_FOR_PERIOD_CONFIDENCE = 3

    def score(self, cnn_probs: np.ndarray, feats: TransitFeatures) -> CandidateVerdict:
        cnn_prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, cnn_probs)}
        evidence: list[EvidenceItem] = []
        fp_flags: list[str] = []

        adjusted = cnn_probs.copy().astype(float)
        planet_idx = CLASS_NAMES.index("PLANET_TRANSIT")
        eb_idx = CLASS_NAMES.index("ECLIPSING_BINARY")

        cnn_planet_conf = float(cnn_probs[planet_idx])

        # Adaptive gating: trust CNN more when confident AND period fold is unreliable.
        # When few transits are observed (<4), BLS may have found the wrong period, so
        # classical vetting on that fold produces noise — don't let noise override CNN.
        # When many transits exist, the fold is reliable and classical tests should fire normally.
        n_trans = feats.n_transits_observed
        if cnn_planet_conf > 0.90 and n_trans < 4:
            sigma_multiplier = 4.0  # CNN confident + unreliable fold → trust CNN
        elif cnn_planet_conf > 0.80 and n_trans < 8:
            sigma_multiplier = 1.8
        elif cnn_planet_conf > 0.60:
            sigma_multiplier = 1.2
        else:
            sigma_multiplier = 1.0

        eff_odd_even_threshold = self.ODD_EVEN_FP_SIGMA * sigma_multiplier
        eff_secondary_threshold = self.SECONDARY_ECLIPSE_FP_SIGMA * sigma_multiplier

        # ---- Odd-even depth test ----
        odd_even_reliable = feats.n_transits_observed >= 5 and feats.snr >= 8.0
        if odd_even_reliable:
            effective_odd_even = feats.odd_even_depth_diff_sigma
        else:
            # Discount unreliable periods, but not so aggressively that huge odd-even sigmas
            # (e.g. 9σ) get suppressed below threshold. Use a floor: even unreliable signals
            # at 8σ+ get at least 0.5× discount rather than 0.3×.
            discount = 0.3 if feats.odd_even_depth_diff_sigma < 8.0 else 0.5
            effective_odd_even = feats.odd_even_depth_diff_sigma * discount

        if effective_odd_even > eff_odd_even_threshold:
            penalty = min(0.7, 0.15 * effective_odd_even)
            shift = adjusted[planet_idx] * penalty
            adjusted[planet_idx] -= shift
            adjusted[eb_idx] += shift
            fp_flags.append("odd_even_depth_mismatch")
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                f"Odd- and even-numbered transits differ in depth by {feats.odd_even_depth_diff_sigma:.1f}σ — "
                f"classic signature of two overlapping eclipse depths from a binary system folded at half its true period.",
                strength=min(1.0, effective_odd_even / 10)))
        else:
            evidence.append(EvidenceItem("supports", "PLANET_TRANSIT",
                f"Odd vs even transit depths are consistent ({feats.odd_even_depth_diff_sigma:.1f}σ difference) — "
                f"no sign of an eclipsing-binary period alias.", strength=0.4))

        # ---- Secondary eclipse test ----
        # Weight by period-recovery confidence: with few observed transits or weak periodicity,
        # the phase fold itself is less trustworthy (possible harmonic aliasing), which can create
        # a spurious "secondary" from a real transit landing at the wrong phase. We require BOTH
        # high statistical significance AND reasonable period confidence before treating this as
        # strong evidence against a planetary origin.
        period_confidence_ok = feats.n_transits_observed >= 3 and feats.periodicity_strength >= 3.0
        effective_sec_sigma = feats.secondary_eclipse_sig_sigma if period_confidence_ok else feats.secondary_eclipse_sig_sigma * 0.5

        if effective_sec_sigma > eff_secondary_threshold:
            penalty = min(0.8, 0.12 * effective_sec_sigma)
            shift = adjusted[planet_idx] * penalty
            adjusted[planet_idx] -= shift
            adjusted[eb_idx] += shift
            fp_flags.append("secondary_eclipse_detected")
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                f"A significant secondary eclipse ({feats.secondary_eclipse_depth_ppm:.0f} ppm, "
                f"{feats.secondary_eclipse_sig_sigma:.1f}σ) was detected near phase 0.5 — planets essentially "
                f"never produce a detectable secondary eclipse at this depth; this points to a stellar companion.",
                strength=min(1.0, effective_sec_sigma / 10)))
        else:
            evidence.append(EvidenceItem("supports", "PLANET_TRANSIT",
                "No significant secondary eclipse detected near phase 0.5, consistent with a planetary companion.",
                strength=0.3))

        # ---- Transit shape (V-shape / grazing) test ----
        if feats.transit_shape_score < self.SHAPE_SCORE_FP_THRESHOLD:
            penalty = 0.3
            shift = adjusted[planet_idx] * penalty
            adjusted[planet_idx] -= shift
            adjusted[eb_idx] += shift
            fp_flags.append("v_shaped_grazing_transit")
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                f"Transit shape is sharply V-shaped (shape score {feats.transit_shape_score:.2f}/1.0) rather than "
                f"flat-bottomed — suggestive of a grazing eclipse rather than a clean planetary transit.",
                strength=0.5))
        elif feats.transit_shape_score > 0.6:
            evidence.append(EvidenceItem("supports", "PLANET_TRANSIT",
                f"Transit shows a flat-bottomed profile (shape score {feats.transit_shape_score:.2f}/1.0), "
                f"consistent with a fully-transiting planetary disk.", strength=0.4))

        # ---- Duration/period ratio sanity check ----
        if feats.duration_period_ratio > 0.25:
            fp_flags.append("anomalous_duration_period_ratio")
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                f"Transit duration is unusually long relative to the orbital period "
                f"(ratio={feats.duration_period_ratio:.2f}) — atypical for a bound planetary orbit at this period.",
                strength=0.3))

        # ---- In-transit scatter (blends/contamination) ----
        if feats.in_transit_std_ratio > 1.6:
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                f"Scatter during transit is {feats.in_transit_std_ratio:.1f}x higher than out-of-transit — "
                f"may indicate blending with a nearby variable source or stellar activity contamination.",
                strength=0.3))

        # ---- SNR / data quality ----
        data_quality_warning = None
        if feats.snr < self.MIN_SNR_RELIABLE:
            data_quality_warning = (f"Detection SNR is low ({feats.snr:.1f}); confidence is reduced and this "
                                      f"candidate would benefit from additional observations.")
            adjusted *= 0.85  # flatten all probabilities toward uncertainty
            adjusted += (1 - adjusted.sum()) / len(adjusted)  # renormalize-ish softening

        if feats.n_transits_observed < self.MIN_TRANSITS_FOR_PERIOD_CONFIDENCE:
            warn2 = (f"Only {feats.n_transits_observed} transit(s) observed in the baseline — the orbital "
                     f"period estimate carries higher uncertainty (possible aliasing) until more transits are caught.")
            data_quality_warning = (data_quality_warning + " " + warn2) if data_quality_warning else warn2

        # ---- periodicity strength as a positive/negative signal for INSTRUMENT_NOISE ----
        noise_idx = CLASS_NAMES.index("INSTRUMENT_NOISE")
        if feats.periodicity_strength < 2.0 and feats.snr < self.MIN_SNR_RELIABLE:
            shift = min(0.3, adjusted[planet_idx] * 0.2)
            adjusted[planet_idx] -= shift
            adjusted[noise_idx] += shift
            evidence.append(EvidenceItem("against", "PLANET_TRANSIT",
                "The detected periodicity is weak relative to the background power spectrum, raising the "
                "possibility this is a noise/systematics artifact rather than a real periodic signal.",
                strength=0.3))

        adjusted = np.clip(adjusted, 1e-6, None)
        adjusted = adjusted / adjusted.sum()

        final_idx = int(np.argmax(adjusted))
        final_label = CLASS_NAMES[final_idx]
        final_confidence = float(adjusted[final_idx])

        is_fp = (final_label == "PLANET_TRANSIT" and len(fp_flags) >= 2) or \
                (final_label != "PLANET_TRANSIT" and cnn_prob_dict.get("PLANET_TRANSIT", 0) > 0.3)

        return CandidateVerdict(
            final_label=final_label,
            final_confidence=final_confidence,
            cnn_probabilities=cnn_prob_dict,
            rule_adjusted_probabilities={name: float(p) for name, p in zip(CLASS_NAMES, adjusted)},
            evidence=evidence,
            false_positive_flags=fp_flags,
            is_likely_false_positive=is_fp,
            data_quality_warning=data_quality_warning,
        )
