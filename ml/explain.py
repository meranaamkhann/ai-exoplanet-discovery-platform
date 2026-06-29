"""
explain.py
===========
Explainable AI layer: produces (1) saliency maps over the phase-folded light
curve showing which regions drove the CNN's decision, and (2) natural-
language scientific explanations combining the CNN's evidence with the
classical feature evidence from ensemble.py.

Saliency approach: Integrated Gradients computed w.r.t. the global/local view
inputs. This highlights which phase bins most influenced the predicted class
— directly visualizable as a heatmap overlay on the light curve.
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from model_cnn import ExoplanetCNN, CLASS_NAMES
from ensemble import CandidateVerdict
from features import TransitFeatures


@dataclass
class SaliencyResult:
    global_saliency: np.ndarray   # same length as global_view, normalized 0-1
    local_saliency: np.ndarray    # same length as local_view, normalized 0-1
    aux_importance: dict           # {feature_name: importance} for the aux features


def compute_saliency(model: ExoplanetCNN, global_view: torch.Tensor, local_view: torch.Tensor,
                       aux: torch.Tensor, target_class: int, aux_feature_names: list[str],
                       n_integration_steps: int = 20) -> SaliencyResult:
    """
    Integrated Gradients (Sundararajan et al. 2017): integrates gradients along
    a straight-line path from a baseline (zeros = "flat, no-signal" curve) to
    the actual input. More faithful than raw gradients and still fast enough
    for real-time use (no GPU needed at this scale).
    """
    model.eval()
    g = global_view.unsqueeze(0).clone().requires_grad_(False)
    l = local_view.unsqueeze(0).clone().requires_grad_(False)
    a = aux.unsqueeze(0).clone().requires_grad_(False)

    baseline_g = torch.zeros_like(g)
    baseline_l = torch.zeros_like(l)
    baseline_a = torch.zeros_like(a)

    total_grad_g = torch.zeros_like(g)
    total_grad_l = torch.zeros_like(l)
    total_grad_a = torch.zeros_like(a)

    for step in range(1, n_integration_steps + 1):
        alpha = step / n_integration_steps
        gi = (baseline_g + alpha * (g - baseline_g)).clone().requires_grad_(True)
        li = (baseline_l + alpha * (l - baseline_l)).clone().requires_grad_(True)
        ai = (baseline_a + alpha * (a - baseline_a)).clone().requires_grad_(True)

        logits = model(gi, li, ai)
        target_logit = logits[0, target_class]
        grads = torch.autograd.grad(target_logit, [gi, li, ai], retain_graph=False)
        total_grad_g += grads[0]
        total_grad_l += grads[1]
        total_grad_a += grads[2]

    avg_grad_g = total_grad_g / n_integration_steps
    avg_grad_l = total_grad_l / n_integration_steps
    avg_grad_a = total_grad_a / n_integration_steps

    ig_g = ((g - baseline_g) * avg_grad_g).squeeze(0).detach().numpy()
    ig_l = ((l - baseline_l) * avg_grad_l).squeeze(0).detach().numpy()
    ig_a = ((a - baseline_a) * avg_grad_a).squeeze(0).detach().numpy()

    def normalize(x):
        x = np.abs(x)
        if x.max() > 1e-10:
            x = x / x.max()
        return x

    global_sal = normalize(ig_g)
    local_sal = normalize(ig_l)
    aux_imp = {name: float(v) for name, v in zip(aux_feature_names, np.abs(ig_a))}

    return SaliencyResult(global_saliency=global_sal, local_saliency=local_sal, aux_importance=aux_imp)


def generate_scientific_explanation(verdict: CandidateVerdict, feats: TransitFeatures,
                                      target_name: str = "the target star") -> str:
    """
    Produces a natural-language, research-assistant-style explanation of the
    detection, combining the quantitative verdict with plain-English framing
    a scientist could paste directly into observation notes.
    """
    label_readable = {
        "PLANET_TRANSIT": "a planetary transit candidate",
        "ECLIPSING_BINARY": "an eclipsing binary star system",
        "STELLAR_VARIABILITY": "intrinsic stellar variability (pulsation)",
        "STARSPOT_ACTIVITY": "rotational modulation from starspot activity",
        "INSTRUMENT_NOISE": "instrumental noise / systematics with no clear astrophysical periodic signal",
    }[verdict.final_label]

    lines = []
    lines.append(
        f"Analysis of {target_name} identifies a periodic signal at P = {feats.period_days:.4f} days "
        f"(transit duration {feats.duration_hours:.2f} h, depth {feats.depth_ppm:.0f} ppm, "
        f"detection SNR = {feats.snr:.1f}). The pipeline classifies this signal as {label_readable}, "
        f"with {verdict.final_confidence*100:.1f}% calibrated confidence."
    )

    if verdict.final_label == "PLANET_TRANSIT":
        lines.append(
            f"If confirmed, the implied planetary radius is approximately "
            f"{feats.planet_radius_re_estimate:.2f} Earth radii, based on the measured transit depth and an "
            f"assumed/measured stellar radius. {feats.n_transits_observed} transit event(s) were observed "
            f"within the baseline."
        )

    supporting = [e for e in verdict.evidence if e.direction == "supports"]
    against = [e for e in verdict.evidence if e.direction == "against"]

    if supporting:
        lines.append("Supporting evidence: " + " ".join(e.reason for e in supporting[:3]))
    if against:
        lines.append("Caveats / contradicting evidence: " + " ".join(e.reason for e in against[:3]))

    if verdict.is_likely_false_positive:
        lines.append(
            "⚠ This candidate is flagged as a probable false positive based on the vetting tests above "
            "(" + ", ".join(verdict.false_positive_flags) + "). Manual review is recommended before "
            "further follow-up resources are allocated."
        )

    if verdict.data_quality_warning:
        lines.append("Data quality note: " + verdict.data_quality_warning)

    return " ".join(lines)


def generate_observation_summary(verdicts: list[tuple[str, CandidateVerdict, TransitFeatures]]) -> str:
    """Summarize a batch of analyzed targets — for the 'automated observation summary' feature."""
    n = len(verdicts)
    n_planet = sum(1 for _, v, _ in verdicts if v.final_label == "PLANET_TRANSIT" and not v.is_likely_false_positive)
    n_fp = sum(1 for _, v, _ in verdicts if v.is_likely_false_positive)
    n_eb = sum(1 for _, v, _ in verdicts if v.final_label == "ECLIPSING_BINARY")
    n_var = sum(1 for _, v, _ in verdicts if v.final_label in ("STELLAR_VARIABILITY", "STARSPOT_ACTIVITY"))
    n_noise = sum(1 for _, v, _ in verdicts if v.final_label == "INSTRUMENT_NOISE")

    high_conf_candidates = sorted(
        [(name, v, f) for name, v, f in verdicts if v.final_label == "PLANET_TRANSIT" and not v.is_likely_false_positive],
        key=lambda x: -x[1].final_confidence
    )[:5]

    lines = [
        f"Batch analysis of {n} target(s) complete. "
        f"{n_planet} promising planet transit candidate(s) identified, {n_eb} likely eclipsing binaries, "
        f"{n_var} variable/active stars, {n_noise} dominated by instrumental noise, and {n_fp} flagged as "
        f"probable false positives requiring manual review."
    ]
    if high_conf_candidates:
        lines.append("Top candidates by confidence:")
        for name, v, f in high_conf_candidates:
            lines.append(f"  • {name}: P={f.period_days:.3f}d, depth={f.depth_ppm:.0f}ppm, "
                          f"confidence={v.final_confidence*100:.1f}%, est. radius={f.planet_radius_re_estimate:.2f} R⊕")
    return "\n".join(lines)
