BOX_COORDINATE_POLAR = "polar"
BOX_COORDINATE_CARTESIAN = "cartesian"
BOX_COORDINATE_CHOICES = (
    BOX_COORDINATE_POLAR,
    BOX_COORDINATE_CARTESIAN,
)

EVAL_COORDINATE_AUTO = "auto"
EVAL_COORDINATE_BOTH = "both"
EVAL_COORDINATE_CHOICES = (
    EVAL_COORDINATE_AUTO,
    BOX_COORDINATE_POLAR,
    BOX_COORDINATE_CARTESIAN,
    EVAL_COORDINATE_BOTH,
)


def validate_box_coordinate_mode(value):
    normalized = str(value).strip().lower()
    if normalized not in BOX_COORDINATE_CHOICES:
        raise ValueError(
            "box_coordinate_mode must be one of "
            f"{BOX_COORDINATE_CHOICES}, got {value!r}"
        )
    return normalized


def validate_eval_coordinate_mode(value):
    normalized = str(value).strip().lower()
    if normalized not in EVAL_COORDINATE_CHOICES:
        raise ValueError(
            "eval_coordinate_mode must be one of "
            f"{EVAL_COORDINATE_CHOICES}, got {value!r}"
        )
    return normalized


def resolve_evaluation_coordinate_mode(
        eval_coordinate_mode,
        box_coordinate_mode,
    ):
    """Resolve standalone metric geometry from the checkpoint box geometry."""
    requested_mode = validate_eval_coordinate_mode(eval_coordinate_mode)
    box_mode = validate_box_coordinate_mode(box_coordinate_mode)

    if requested_mode == EVAL_COORDINATE_AUTO:
        effective_mode = box_mode
    else:
        effective_mode = requested_mode

    if (
        effective_mode in BOX_COORDINATE_CHOICES
        and effective_mode != box_mode
    ):
        raise ValueError(
            f"eval_coordinate_mode={requested_mode!r} does not match the "
            f"checkpoint box_coordinate_mode={box_mode!r}. Use 'auto' for "
            "the direct metric, or 'both' to request converted auxiliary metrics."
        )

    official_eval_enabled = effective_mode in (
        BOX_COORDINATE_CARTESIAN,
        EVAL_COORDINATE_BOTH,
    )
    polar_eval_enabled = effective_mode in (
        BOX_COORDINATE_POLAR,
        EVAL_COORDINATE_BOTH,
    )
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "official_eval_enabled": official_eval_enabled,
        "polar_eval_enabled": polar_eval_enabled,
        "primary_geometry": box_mode,
        "official_geometry_source": (
            "direct"
            if box_mode == BOX_COORDINATE_CARTESIAN
            else "converted_auxiliary"
        ),
        "polar_geometry_source": (
            "direct"
            if box_mode == BOX_COORDINATE_POLAR
            else "converted_auxiliary"
        ),
    }
