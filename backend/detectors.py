"""Explicit algorithm selection. Refined is the default detector."""

DETECTOR_NAMES = ("baseline", "contact", "fusion", "refined")


def detect(image, aorta_mask, *, detector: str = "refined", parameters: dict | None = None):
    """Run one independent detector using its default settings."""
    from backend.configuration import configuration, option_types
    resolved = configuration(detector, parameters)
    options = {name: cls(**resolved["parameters"][name]) for name, cls in option_types(detector).items()}
    if detector == "baseline":
        from backend.detection import detect_daughters
        result = detect_daughters(image, aorta_mask, options["baseline"])
    elif detector == "contact":
        from backend.contact_detection import detect_contacts
        result = detect_contacts(image, aorta_mask, options["contact"])
    elif detector == "fusion":
        from backend.fusion_detection import detect_fusion
        result = detect_fusion(image, aorta_mask, options["fusion"],
                               baseline_options=options["baseline"], contact_options=options["contact"])
    elif detector == "refined":
        from backend.refined_detection import detect_refined
        result = detect_refined(image, aorta_mask, options["refined"], fusion_options=options["fusion"],
                                baseline_options=options["baseline"], contact_options=options["contact"])
    else:
        raise ValueError(f"Unknown detector {detector!r}; choose from {DETECTOR_NAMES}")
    result.diagnostics["detector"] = detector
    result.diagnostics["configuration"] = resolved
    return result
