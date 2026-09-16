BOX_COORDINATE_POLAR = "polar"
BOX_COORDINATE_CARTESIAN = "cartesian"
BOX_COORDINATE_CHOICES = (
    BOX_COORDINATE_POLAR,
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
