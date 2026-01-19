"""Tests for layout packing and unpacking.

This module tests the even-odd (checkerboard) layout system.
"""

import torch

import plaq as pq


def test_pack_unpack_eo_roundtrip() -> None:
    """Test that pack_eo and unpack_eo are exact inverses."""
    lat = pq.Lattice((4, 4, 4, 8))
    dtype = torch.complex128
    device = torch.device("cpu")

    # Create random site tensor
    psi_site = torch.randn(lat.volume, 4, 3, dtype=dtype, device=device)

    # Pack to eo
    psi_eo = pq.pack_eo(psi_site, lat)

    # Unpack back to site
    psi_site2 = pq.unpack_eo(psi_eo, lat)

    # Should be exactly equal
    assert torch.allclose(psi_site, psi_site2, atol=0, rtol=0), "pack_eo/unpack_eo roundtrip failed"


def test_pack_eo_shape() -> None:
    """Test that packed tensor has correct shape."""
    lat = pq.Lattice((4, 4, 4, 8))
    dtype = torch.complex128
    device = torch.device("cpu")

    psi_site = torch.randn(lat.volume, 4, 3, dtype=dtype, device=device)
    psi_eo = pq.pack_eo(psi_site, lat)

    expected_shape = (2, lat.volume // 2, 4, 3)
    assert psi_eo.shape == expected_shape, f"Expected shape {expected_shape}, got {psi_eo.shape}"


def test_pack_eo_parity_correctness() -> None:
    """Test that pack_eo correctly separates even and odd sites."""
    lat = pq.Lattice((4, 4, 4, 4))
    dtype = torch.complex128
    device = torch.device("cpu")

    # Create a tensor where each site has a unique value equal to its parity
    # This lets us verify the packing is correct
    psi_site = torch.zeros(lat.volume, 4, 3, dtype=dtype, device=device)

    for site_id in range(lat.volume):
        parity = lat.parity(site_id)
        # Mark even sites with 1.0, odd sites with 2.0
        psi_site[site_id, :, :] = float(parity + 1)

    psi_eo = pq.pack_eo(psi_site, lat)

    # All even sites (index 0) should have value 1.0
    even_values = psi_eo[0, :, :, :].real
    assert torch.allclose(even_values, torch.ones_like(even_values)), (
        "Even sites not correctly packed"
    )

    # All odd sites (index 1) should have value 2.0
    odd_values = psi_eo[1, :, :, :].real
    assert torch.allclose(odd_values, 2.0 * torch.ones_like(odd_values)), (
        "Odd sites not correctly packed"
    )


def test_even_odd_site_counts() -> None:
    """Test that even and odd site counts are correct."""
    # For a lattice with all even dimensions, we have V/2 even and V/2 odd sites
    lat = pq.Lattice((4, 4, 4, 8))

    assert len(lat.even_sites) == lat.volume // 2
    assert len(lat.odd_sites) == lat.volume // 2
    assert len(lat.even_sites) + len(lat.odd_sites) == lat.volume


def test_even_odd_sites_disjoint() -> None:
    """Test that even and odd site lists are disjoint and complete."""
    lat = pq.Lattice((4, 4, 4, 8))

    all_sites = set(range(lat.volume))
    even_set = set(lat.even_sites.tolist())
    odd_set = set(lat.odd_sites.tolist())

    # Should be disjoint
    assert even_set.isdisjoint(odd_set), "Even and odd sites overlap"

    # Should cover all sites
    assert even_set | odd_set == all_sites, "Even and odd don't cover all sites"


def test_spinor_field_layout_conversion() -> None:
    """Test SpinorField layout conversion preserves data."""
    lat = pq.Lattice((4, 4, 4, 8))

    # Create field in site layout
    psi_site = pq.SpinorField.random(lat, layout="site")
    original_data = psi_site.site.clone()

    # Convert to eo and back
    psi_eo = psi_site.as_layout("eo")
    psi_back = psi_eo.as_layout("site")

    assert torch.allclose(original_data, psi_back.site), "SpinorField layout roundtrip failed"


def test_spinor_field_caching() -> None:
    """Test that SpinorField caches layout conversions."""
    lat = pq.Lattice((4, 4, 4, 8))

    psi = pq.SpinorField.random(lat, layout="site")

    # First access to eo should cache
    eo1 = psi.eo

    # Second access should return same object
    eo2 = psi.eo

    assert eo1 is eo2, "EO cache not working"

    # Site should still be cached
    site1 = psi.site
    site2 = psi.site
    assert site1 is site2, "Site cache not working"
