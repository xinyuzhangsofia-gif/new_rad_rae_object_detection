BOX_COORDINATE_POLAR = "polar"
BOX_COORDINATE_CARTESIAN = "cartesian"
BOX_COORDINATE_CHOICES = (
    BOX_COORDINATE_POLAR,
    BOX_COORDINATE_CARTESIAN,
)

EVAL_COORDINATE_AUTO = "auto"
EVAL_COORDINATE_CHOICES = (
    EVAL_COORDINATE_AUTO,
    BOX_COORDINATE_CARTESIAN,
)


def validate_box_coordinate_mode(value):
    normalized = str(value).strip().lower()
    if normalized not in BOX_COORDINATE_CHOICES:
        raise ValueError(
            "box_coordinate_mode must be one of "
            f"{BOX_COORDINATE_CHOICES}, got {value!r}"
        )
    return normalized


def require_cartesian_data(value=BOX_COORDINATE_CARTESIAN):
    """Reject retired Polar GT/checkpoints at data-consuming entry points.

    Polar geometry remains available for internal RAE grids and visualization;
    it is no longer a supported annotation input format.
    """
    normalized = validate_box_coordinate_mode(value)
    if normalized != BOX_COORDINATE_CARTESIAN:
        raise ValueError(
            "Only Cartesian GT/checkpoints are supported by the data pipeline; "
            "Polar input is no longer supported. Use radar-aligned Cartesian "
            "labels and a Cartesian checkpoint, not a coordinate override."
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
    box_mode = require_cartesian_data(box_coordinate_mode)

    if requested_mode == EVAL_COORDINATE_AUTO:
        effective_mode = box_mode
    else:
        effective_mode = requested_mode

    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "official_eval_enabled": True,
        "primary_geometry": box_mode,
        "official_geometry_source": "direct",
    }
