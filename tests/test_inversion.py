"""Unit tests for temperature inversion detection in inversion.py.

All tests use synthetic profiles with hand-verifiable values.
No external or data files are used.
"""

import numpy as np
import pytest

from inversion import InversionLayer, find_inversions, strongest_inversion


def test_standard_lapse_rate_no_inversion():
    """1. Standard lapse-rate profile (temp falls with height) returns no inversion."""
    heights = [0, 100, 200, 300, 400]
    temps = [20.0, 19.0, 18.0, 17.0, 16.0]

    layers = find_inversions(heights, temps)
    assert layers == []
    assert strongest_inversion(heights, temps) is None


def test_surface_based_inversion():
    """2. Surface-based inversion: lowest level starts an increasing layer.

    Heights: 0, 100, 200, 300, 400
    Temps:   8,  10,  12,  11,   9
    -> base 0, top 200, depth 200, strength 4, type "surface"
    """
    heights = [0, 100, 200, 300, 400]
    temps = [8.0, 10.0, 12.0, 11.0, 9.0]

    layers = find_inversions(heights, temps)
    assert len(layers) == 1

    inv = layers[0]
    assert inv.base_height == 0.0
    assert inv.top_height == 200.0
    assert inv.depth_m == 200.0
    assert inv.strength_c == 4.0
    assert inv.kind == "surface"
    assert inv.type == "surface"


def test_elevated_inversion_with_normal_layer_below():
    """3. Elevated inversion with a normal lapse-rate layer below it.

    Heights: 0,   100,  200,  300,  400,  500
    Temps:   20,   18,   17,   19,   21,   16
    -> base 200, top 400, depth 200, strength 4, type "elevated"
    """
    heights = [0, 100, 200, 300, 400, 500]
    temps = [20.0, 18.0, 17.0, 19.0, 21.0, 16.0]

    layers = find_inversions(heights, temps)
    assert len(layers) == 1

    inv = layers[0]
    assert inv.base_height == 200.0
    assert inv.top_height == 400.0
    assert inv.depth_m == 200.0
    assert inv.strength_c == 4.0
    assert inv.kind == "elevated"
    assert inv.type == "elevated"


def test_two_separate_inversions_and_strongest():
    """4. Two separate inversions: returns both, and helper returns the strongest.

    Heights: 0,   100,  200,  300,  400,  500,  600,  700
    Temps:   10,   14,   12,   11,   15,   18,   14,   12
    -> Layer 1: base 0, top 100, depth 100, strength 4, type "surface"
    -> Layer 2: base 300, top 500, depth 200, strength 7, type "elevated"
    -> Strongest: Layer 2 (strength 7)
    """
    heights = [0, 100, 200, 300, 400, 500, 600, 700]
    temps = [10.0, 14.0, 12.0, 11.0, 15.0, 18.0, 14.0, 12.0]

    layers = find_inversions(heights, temps)
    assert len(layers) == 2

    l1, l2 = layers
    # Layer 1: surface-based
    assert l1.base_height == 0.0
    assert l1.top_height == 100.0
    assert l1.depth_m == 100.0
    assert l1.strength_c == 4.0
    assert l1.kind == "surface"

    # Layer 2: elevated
    assert l2.base_height == 300.0
    assert l2.top_height == 500.0
    assert l2.depth_m == 200.0
    assert l2.strength_c == 7.0
    assert l2.kind == "elevated"

    # Strongest inversion helper
    best_from_layers = strongest_inversion(layers)
    assert best_from_layers == l2
    assert best_from_layers.strength_c == 7.0

    best_from_arrays = strongest_inversion(heights, temps)
    assert best_from_arrays == l2


def test_small_noise_within_tolerance_does_not_split():
    """5. Small noise: a dip of at most tol_c (default 0.2) does not split an increasing layer.

    Heights: 0,    100,   200,   300,   400,   500
    Temps:   10.0, 12.0,  11.9,  13.0,  11.0,  9.0
    Dip at 200m is 12.0 - 11.9 = 0.1 <= 0.2 (tol_c).
    With tol_c=0.2: layer spans 0 to 300m (strength 3.0).
    With tol_c=0.05: dip of 0.1 exceeds tol_c and splits the layer.
    """
    heights = [0, 100, 200, 300, 400, 500]
    temps = [10.0, 12.0, 11.9, 13.0, 11.0, 9.0]

    # tol_c = 0.2 (default) -> does not split
    layers = find_inversions(heights, temps, tol_c=0.2)
    assert len(layers) == 1
    assert layers[0].base_height == 0.0
    assert layers[0].top_height == 300.0
    assert layers[0].depth_m == 300.0
    assert layers[0].strength_c == 3.0

    # tol_c = 0.05 -> dip of 0.1 terminates the first layer at 100m
    strict_layers = find_inversions(heights, temps, tol_c=0.05)
    assert len(strict_layers) == 2
    assert strict_layers[0].base_height == 0.0
    assert strict_layers[0].top_height == 100.0
    assert strict_layers[0].strength_c == 2.0


def test_isothermal_layer_not_inversion():
    """6. Isothermal layer (equal temps) does not count as an inversion."""
    heights = [0, 100, 200, 300]

    # Perfectly flat profile
    temps_flat = [15.0, 15.0, 15.0, 15.0]
    assert find_inversions(heights, temps_flat) == []

    # Flat followed by falling
    temps_falling = [15.0, 15.0, 14.0, 13.0]
    assert find_inversions(heights, temps_falling) == []


def test_bad_input_handling():
    """7. Bad input handling:

    - NaN values are dropped pairwise.
    - Unsorted heights raise ValueError.
    - Fewer than 2 valid points returns no inversion ([]).
    """
    # NaNs dropped pairwise
    heights_nan = [0, 100, np.nan, 200, 300, 400]
    temps_nan = [8.0, 10.0, 99.0, 12.0, 11.0, 9.0]
    layers = find_inversions(heights_nan, temps_nan)
    assert len(layers) == 1
    assert layers[0].base_height == 0.0
    assert layers[0].top_height == 200.0
    assert layers[0].strength_c == 4.0

    # NaN in temp
    heights_t_nan = [0, 100, 200, 300, 400]
    temps_t_nan = [8.0, np.nan, 12.0, 11.0, 9.0]
    # Remaining valid: (0, 8), (200, 12), (300, 11), (400, 9)
    layers_t = find_inversions(heights_t_nan, temps_t_nan)
    assert len(layers_t) == 1
    assert layers_t[0].base_height == 0.0
    assert layers_t[0].top_height == 200.0
    assert layers_t[0].strength_c == 4.0

    # Unsorted heights (decreasing) -> ValueError
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        find_inversions([0, 200, 100], [10.0, 12.0, 14.0])

    # Unsorted heights (duplicate height) -> ValueError
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        find_inversions([0, 100, 100, 200], [10.0, 11.0, 12.0, 13.0])

    # Fewer than 2 valid points -> []
    assert find_inversions([], []) == []
    assert find_inversions([100], [15.0]) == []
    assert find_inversions([100, np.nan], [15.0, 20.0]) == []
    assert find_inversions([np.nan, np.nan], [np.nan, np.nan]) == []


def test_min_depth_filter():
    """Verify min_depth_m parameter filters shallow layers."""
    heights = [0, 50, 100, 200, 300]
    temps = [10.0, 12.0, 11.0, 9.0, 8.0]
    # Layer: base 0 to top 50, depth 50m, strength 2.0C

    # With min_depth_m = 0.0 -> kept
    assert len(find_inversions(heights, temps, min_depth_m=0.0)) == 1

    # With min_depth_m = 100.0 -> filtered out
    assert find_inversions(heights, temps, min_depth_m=100.0) == []
