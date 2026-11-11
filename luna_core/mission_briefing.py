"""
Generates the full mission-style text briefing for the Mission Briefing tab —
several paragraphs, not a one-liner.
"""


def generate_briefing(result: dict) -> str:
    stats = result["class_stats"]
    risk = result["risk_breakdown"]
    score = result["score"]
    verdict = result["verdict"]
    confidence = result["confidence_pct"]

    lines = []
    lines.append("LUNATWIN AI \u2014 LANDING SITE MISSION BRIEFING")
    lines.append("=" * 46)
    lines.append("")
    lines.append(
        "ENHANCEMENT SUMMARY\n"
        "The source image was denoised, contrast-enhanced with CLAHE, and gamma-corrected "
        "to recover detail in shadowed regions before analysis. This step materially "
        "improves detection reliability on low-light lunar imagery."
    )
    lines.append("")
    lines.append(
        "HAZARDS DETECTED\n"
        f"- Crater coverage: {stats['crater']['area_pct']}% of frame "
        f"({risk['crater_risk']['level']} risk \u2014 {risk['crater_risk']['reason']})\n"
        f"- Boulder coverage: {stats['boulder']['area_pct']}% of frame "
        f"({risk['boulder_risk']['level']} risk \u2014 {risk['boulder_risk']['reason']})\n"
        f"- Rough terrain coverage: {stats['rough_terrain']['area_pct']}% of frame "
        f"({risk['roughness_risk']['level']} risk \u2014 {risk['roughness_risk']['reason']})\n"
        f"- Smooth/safe terrain: {stats['smooth_safe']['area_pct']}% of frame."
    )
    lines.append("")
    lines.append(
        "CONFIDENCE SUMMARY\n"
        f"Average model confidence across the frame is {confidence}%, derived from "
        "per-pixel softmax entropy of the segmentation output \u2014 regions where the "
        "model's class probabilities are spread out (uncertain) score lower here, "
        "independent of what hazard class was predicted."
    )
    lines.append("")
    lines.append(
        "LANDING ASSESSMENT\n"
        f"Landing Score: {score}/100\n"
        f"Verdict: {verdict}\n"
        + {
            "Safe": "The majority of the frame is classified smooth/safe with low hazard density. "
                    "No major obstacles dominate the highest-confidence regions.",
            "Caution": "Hazard density is moderate. Safe zones exist but are interspersed with "
                       "crater, boulder, or rough-terrain cells \u2014 a landing site should be chosen "
                       "from the lowest-hazard cells specifically, not the frame average.",
            "Risky": "Hazard density is high across a significant portion of the frame. This site "
                     "is not recommended without further imaging or a narrower landing ellipse "
                     "targeted at the few low-hazard cells identified.",
        }[verdict]
    )
    lines.append("")
    lines.append(
        "RECOMMENDATION\n"
        + {
            "Safe": "Proceed with this landing zone. Prioritize cells with both low hazard score "
                    "and high confidence for final touchdown targeting.",
            "Caution": "Proceed with a refined targeting pass \u2014 use the Landing Zone grid to "
                       "select specific low-hazard, high-confidence cells rather than the site as a whole.",
            "Risky": "Do not proceed with this site as imaged. Recommend acquiring a follow-up image "
                     "or selecting an alternate candidate site.",
        }[verdict]
    )
    return "\n\n".join(lines)
