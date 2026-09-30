"""Atmospheric Temperature Inversion Analysis for VAAYU.

Provides detection and characterization of temperature inversion layers
(surface-based and elevated) from vertical sounding profiles of height and
temperature. No hard-coded significance thresholds are enforced; minimum
layer depth (`min_depth_m`) and noise tolerance (`tol_c`) are configurable
parameters.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
import numpy as np


@dataclass(frozen=True)
class InversionLayer:
    """Represents a single atmospheric temperature inversion layer.

    Attributes:
        base_height: Altitude of the layer base (m above ground).
        top_height: Altitude of the layer top / peak temperature (m above ground).
        depth_m: Vertical thickness of the inversion layer (top - base) in meters.
        strength_c: Temperature difference across the layer (T_top - T_base) in °C.
        kind: Classification of the inversion layer ('surface' or 'elevated').
    """

    base_height: float
    top_height: float
    depth_m: float
    strength_c: float
    kind: str

    @property
    def type(self) -> str:
        """Alias for kind ('surface' or 'elevated')."""
        return self.kind


def find_inversions(
    height_m: Sequence[float],
    temp_c: Sequence[float],
    tol_c: float = 0.2,
    min_depth_m: float = 0.0,
) -> List[InversionLayer]:
    """Identify atmospheric temperature inversion layers from a vertical sounding profile.

    An inversion layer is defined as a contiguous vertical layer where temperature
    increases with height. The top of the layer is the altitude of the temperature
    maximum before temperature starts falling again.

    Parameters:
        height_m: Heights above ground level (m), ordered from surface upward.
        temp_c: Temperatures (°C) at the corresponding vertical levels.
        tol_c: Noise tolerance (°C). A temperature dip of at most `tol_c` within
            an increasing layer does not split it (default: 0.2 °C).
        min_depth_m: Minimum vertical depth threshold (m). Layers with thickness
            less than `min_depth_m` are filtered out (default: 0.0 m, retaining all
            detected non-zero layers).

    Returns:
        List of InversionLayer objects detected in the profile, ordered from
        lowest to highest base height.

    Raises:
        ValueError: If height_m is not strictly monotonically increasing after
            dropping invalid/NaN entries.
    """
    h = np.asarray(height_m, dtype=float)
    t = np.asarray(temp_c, dtype=float)

    if h.ndim != 1 or t.ndim != 1 or len(h) != len(t):
        raise ValueError("height_m and temp_c must be 1-dimensional arrays of identical length.")

    # Drop NaNs pairwise
    valid = ~(np.isnan(h) | np.isnan(t))
    h = h[valid]
    t = t[valid]

    # Fewer than 2 valid points cannot define an inversion layer
    if len(h) < 2:
        return []

    # Check that heights are strictly monotonically increasing
    diffs = np.diff(h)
    if np.any(diffs <= 0):
        raise ValueError("height_m must be strictly monotonically increasing from the surface upward.")

    surface_height = float(h[0])
    n = len(h)
    layers: List[InversionLayer] = []

    i = 0
    while i < n - 1:
        # An inversion layer begins where temperature increases with height
        if t[i + 1] > t[i]:
            base_idx = i
            peak_idx = i + 1
            peak_temp = t[peak_idx]

            j = i + 2
            while j < n:
                if t[j] > peak_temp:
                    # New local temperature maximum
                    peak_temp = t[j]
                    peak_idx = j
                    j += 1
                else:
                    # Check if the drop is within noise tolerance tol_c
                    drop = peak_temp - t[j]
                    if drop <= tol_c:
                        # Minor fluctuation within tolerance; continue layer
                        j += 1
                    else:
                        # Drop exceeds tolerance; inversion layer has ended
                        break

            base_h = float(h[base_idx])
            top_h = float(h[peak_idx])
            depth = top_h - base_h
            strength = float(peak_temp - t[base_idx])

            # Retain layer if it meets minimum depth and represents positive warming
            if depth >= min_depth_m and depth > 0 and strength > 0:
                kind = "surface" if np.isclose(base_h, surface_height) else "elevated"
                layers.append(
                    InversionLayer(
                        base_height=base_h,
                        top_height=top_h,
                        depth_m=depth,
                        strength_c=strength,
                        kind=kind,
                    )
                )

            # Advance search index to peak_idx to locate subsequent inversions
            i = peak_idx
        else:
            i += 1

    return layers


def strongest_inversion(
    inversions_or_height: Union[Sequence[InversionLayer], Sequence[float]],
    temp_c: Optional[Sequence[float]] = None,
    tol_c: float = 0.2,
    min_depth_m: float = 0.0,
) -> Optional[InversionLayer]:
    """Return the strongest inversion layer (maximum strength_c) from a list or profile.

    Can be invoked either with a pre-computed list of InversionLayer objects:
        strongest_inversion(layers)
    or directly with sounding profile coordinates:
        strongest_inversion(height_m, temp_c, tol_c=0.2, min_depth_m=0.0)

    Returns:
        The InversionLayer with the largest strength_c, or None if no inversions exist.
    """
    if temp_c is not None:
        layers = find_inversions(
            inversions_or_height,  # type: ignore
            temp_c,
            tol_c=tol_c,
            min_depth_m=min_depth_m,
        )
    elif isinstance(inversions_or_height, (list, tuple)) and (
        len(inversions_or_height) == 0 or isinstance(inversions_or_height[0], InversionLayer)
    ):
        layers = list(inversions_or_height)  # type: ignore
    else:
        layers = find_inversions(
            inversions_or_height,  # type: ignore
            temp_c,  # type: ignore
            tol_c=tol_c,
            min_depth_m=min_depth_m,
        )

    if not layers:
        return None

    return max(layers, key=lambda layer: layer.strength_c)
