"""Explicit algorithm selection. The original baseline remains the default."""

DETECTOR_NAMES = ("baseline", "contact", "fusion")


def detect(image, aorta_mask, *, detector: str = "baseline"):
    """Run one independent detector using its default settings."""
    if detector == "baseline":
        from backend.detection import detect_daughters
        result = detect_daughters(image, aorta_mask)
    elif detector == "contact":
        from backend.contact_detection import detect_contacts
        result = detect_contacts(image, aorta_mask)
    elif detector == "fusion":
        from backend.fusion_detection import detect_fusion
        result = detect_fusion(image, aorta_mask)
    else:
        raise ValueError(f"Unknown detector {detector!r}; choose from {DETECTOR_NAMES}")
    result.diagnostics["detector"] = detector
    return result
