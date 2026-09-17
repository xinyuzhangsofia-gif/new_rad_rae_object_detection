"""Canonical semantic overlay colors for visualization backends."""


_BOX_COLORS = {
    "green": {
        "bgr": (0, 255, 0),
        "rgb": (0.0, 1.0, 0.0),
        "matplotlib": "lime",
    },
    "red": {
        "bgr": (0, 0, 255),
        "rgb": (1.0, 0.0, 0.0),
        "matplotlib": "red",
    },
}


def resolve_box_color(color_name, color_space):
    """Resolve one semantic color for OpenCV, Open3D, or Matplotlib."""
    normalized = str(color_name).strip().lower()
    if normalized not in _BOX_COLORS:
        supported = ", ".join(sorted(_BOX_COLORS))
        raise ValueError(
            f"Unknown box color {color_name!r}; choose one of: {supported}"
        )
    return _BOX_COLORS[normalized][color_space]
