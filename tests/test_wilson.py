"""Tests for the Wilson Dirac operator.

This module tests the Wilson operator properties on small lattices.
"""

import torch

import plaq as pq
from plaq.conventions.gamma_milc import get_gamma5


def test_wilson_gamma5_hermiticity_free_field() -> None:
    """Test gamma5-hermiticity: <phi, M psi> = <gamma5 M gamma5 phi, psi>.

    For the Wilson operator, M^dag = gamma5 M gamma5, which implies:
    <phi, M psi> = <M^dag phi, psi> = <gamma5 M gamma5 phi, psi>
    """
    torch.manual_seed(42)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Identity gauge field (free field)
    U = pq.GaugeField.eye(lat)

    # Random spinors
    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # Compute <phi, M psi>
    M_psi = pq.apply_M(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), M_psi.site.flatten())

    # Compute <gamma5 M gamma5 phi, psi>
    gamma5 = get_gamma5(dtype=phi.dtype, device=phi.device)
    phi_site = phi.site
    g5_phi = torch.einsum("ab,vbc->vac", gamma5, phi_site)
    g5_phi_field = pq.SpinorField(g5_phi, lat)

    M_g5_phi = pq.apply_M(U, g5_phi_field, params, bc)
    g5_M_g5_phi = torch.einsum("ab,vbc->vac", gamma5, M_g5_phi.site)

    rhs = torch.vdot(g5_M_g5_phi.flatten(), psi.site.flatten())

    # Should be equal
    diff = torch.abs(lhs - rhs).item()
    tol = 1e-12
    assert diff < tol, f"Gamma5-hermiticity failed: |lhs - rhs| = {diff}"


def test_apply_Mdag_matches_adjoint_free_field() -> None:
    """Test that vdot(phi, M psi) = vdot(Mdag phi, psi).

    This verifies that apply_Mdag correctly computes the adjoint.
    """
    torch.manual_seed(43)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    U = pq.GaugeField.eye(lat)

    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # <phi, M psi>
    M_psi = pq.apply_M(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), M_psi.site.flatten())

    # <Mdag phi, psi>
    Mdag_phi = pq.apply_Mdag(U, phi, params, bc)
    rhs = torch.vdot(Mdag_phi.site.flatten(), psi.site.flatten())

    diff = torch.abs(lhs - rhs).item()
    tol = 1e-11  # Slightly relaxed to account for floating-point accumulation
    assert diff < tol, f"Mdag adjoint test failed: |lhs - rhs| = {diff}"


def test_apply_MdagM_hermitian() -> None:
    """Test that MdagM is Hermitian: <phi, MdagM psi> = <MdagM phi, psi>."""
    torch.manual_seed(44)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    U = pq.GaugeField.eye(lat)

    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # <phi, MdagM psi>
    MdagM_psi = pq.apply_MdagM(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), MdagM_psi.site.flatten())

    # <MdagM phi, psi>
    MdagM_phi = pq.apply_MdagM(U, phi, params, bc)
    rhs = torch.vdot(MdagM_phi.site.flatten(), psi.site.flatten())

    diff = torch.abs(lhs - rhs).item()
    tol = 1e-11  # Slightly relaxed for MdagM (two operator applications)
    assert diff < tol, f"MdagM hermiticity failed: |lhs - rhs| = {diff}"


def test_apply_M_layout_equivalence() -> None:
    """Test that apply_M gives same result regardless of input layout."""
    torch.manual_seed(45)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    U = pq.GaugeField.eye(lat)
    params = pq.WilsonParams(mass=0.1)

    # Create spinor in site layout
    psi_site = pq.SpinorField.random(lat, layout="site")

    # Convert to eo layout
    psi_eo = psi_site.as_layout("eo")

    # Apply M to both
    result_from_site = pq.apply_M(U, psi_site, params, bc)
    result_from_eo = pq.apply_M(U, psi_eo, params, bc)

    # Compare in site layout
    diff = torch.abs(result_from_site.site - result_from_eo.site).max().item()
    tol = 1e-14
    assert diff < tol, f"Layout equivalence failed: max diff = {diff}"


def test_wilson_gamma5_hermiticity_random_gauge() -> None:
    """Test gamma5-hermiticity with random gauge field.

    For the Wilson operator, M^dag = gamma5 M gamma5, which implies:
    <phi, M psi> = <M^dag phi, psi> = <gamma5 M gamma5 phi, psi>

    This test verifies the property holds even for non-trivial gauge configurations.
    """
    torch.manual_seed(100)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Random gauge field
    U = pq.GaugeField.random(lat)

    # Random spinors
    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # Compute <phi, M psi>
    M_psi = pq.apply_M(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), M_psi.site.flatten())

    # Compute <gamma5 M gamma5 phi, psi>
    gamma5 = get_gamma5(dtype=phi.dtype, device=phi.device)
    phi_site = phi.site
    g5_phi = torch.einsum("ab,vbc->vac", gamma5, phi_site)
    g5_phi_field = pq.SpinorField(g5_phi, lat)

    M_g5_phi = pq.apply_M(U, g5_phi_field, params, bc)
    g5_M_g5_phi = torch.einsum("ab,vbc->vac", gamma5, M_g5_phi.site)

    rhs = torch.vdot(g5_M_g5_phi.flatten(), psi.site.flatten())

    # Should be equal
    diff = torch.abs(lhs - rhs).item()
    tol = 1e-12
    assert diff < tol, f"Gamma5-hermiticity failed with random gauge: |lhs - rhs| = {diff}"


def test_apply_Mdag_matches_adjoint_random_gauge() -> None:
    """Test that vdot(phi, M psi) = vdot(Mdag phi, psi) with random gauge field.

    This verifies that apply_Mdag correctly computes the adjoint even for
    non-trivial gauge configurations.
    """
    torch.manual_seed(101)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Random gauge field
    U = pq.GaugeField.random(lat)

    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # <phi, M psi>
    M_psi = pq.apply_M(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), M_psi.site.flatten())

    # <Mdag phi, psi>
    Mdag_phi = pq.apply_Mdag(U, phi, params, bc)
    rhs = torch.vdot(Mdag_phi.site.flatten(), psi.site.flatten())

    diff = torch.abs(lhs - rhs).item()
    tol = 1e-11  # Slightly relaxed to account for floating-point accumulation
    assert diff < tol, f"Mdag adjoint test failed with random gauge: |lhs - rhs| = {diff}"


def test_apply_MdagM_hermitian_random_gauge() -> None:
    """Test that MdagM is Hermitian with random gauge field.

    Verifies <phi, MdagM psi> = <MdagM phi, psi> for non-trivial gauge configurations.
    """
    torch.manual_seed(102)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Random gauge field
    U = pq.GaugeField.random(lat)

    psi = pq.SpinorField.random(lat)
    phi = pq.SpinorField.random(lat)

    params = pq.WilsonParams(mass=0.1)

    # <phi, MdagM psi>
    MdagM_psi = pq.apply_MdagM(U, psi, params, bc)
    lhs = torch.vdot(phi.site.flatten(), MdagM_psi.site.flatten())

    # <MdagM phi, psi>
    MdagM_phi = pq.apply_MdagM(U, phi, params, bc)
    rhs = torch.vdot(MdagM_phi.site.flatten(), psi.site.flatten())

    diff = torch.abs(lhs - rhs).item()
    tol = 1e-11  # Slightly relaxed for MdagM (two operator applications)
    assert diff < tol, f"MdagM hermiticity failed with random gauge: |lhs - rhs| = {diff}"


def test_apply_M_layout_equivalence_random_gauge() -> None:
    """Test that apply_M gives same result regardless of input layout with random gauge.

    This verifies layout equivalence holds for non-trivial gauge configurations.
    """
    torch.manual_seed(103)
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Random gauge field
    U = pq.GaugeField.random(lat)
    params = pq.WilsonParams(mass=0.1)

    # Create spinor in site layout
    psi_site = pq.SpinorField.random(lat, layout="site")

    # Convert to eo layout
    psi_eo = psi_site.as_layout("eo")

    # Apply M to both
    result_from_site = pq.apply_M(U, psi_site, params, bc)
    result_from_eo = pq.apply_M(U, psi_eo, params, bc)

    # Compare in site layout
    diff = torch.abs(result_from_site.site - result_from_eo.site).max().item()
    tol = 1e-14
    assert diff < tol, f"Layout equivalence failed with random gauge: max diff = {diff}"


def test_wilson_free_field_eigenvalue() -> None:
    """Test Wilson operator on a constant spinor in free field.

    For a constant spinor psi(x) = psi_0 at zero momentum, the Wilson
    operator simplifies and we can check the eigenvalue.
    """
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic BC for simplicity
    lat.build_neighbor_tables(bc)

    U = pq.GaugeField.eye(lat)

    # Create a constant spinor
    psi_site = torch.zeros(lat.volume, 4, 3, dtype=torch.complex128)
    psi_site[:, 0, 0] = 1.0  # Constant value at spin=0, color=0
    psi = pq.SpinorField(psi_site, lat)

    params = pq.WilsonParams(mass=0.1, r=1.0)

    result = pq.apply_M(U, psi, params, bc)

    # For p=0 (constant field) with periodic BC:
    # M psi_0 = (m0 + 4r) psi_0 - (1/2) sum_mu [(r - gamma_mu) + (r + gamma_mu)] psi_0
    #         = (m0 + 4r) psi_0 - (1/2) sum_mu [2r] psi_0
    #         = (m0 + 4r) psi_0 - 4r psi_0
    #         = m0 psi_0
    expected_eigenvalue = params.mass

    # Check the result
    result_val = result.site[0, 0, 0].real.item()
    assert abs(result_val - expected_eigenvalue) < 1e-12, (
        f"Free field eigenvalue mismatch: got {result_val}, expected {expected_eigenvalue}"
    )


def test_wilson_boundary_conditions() -> None:
    """Test that boundary conditions affect the result correctly."""
    lat = pq.Lattice((2, 2, 2, 2))

    # Random spinor
    torch.manual_seed(42)
    psi = pq.SpinorField.random(lat)

    U = pq.GaugeField.eye(lat)
    params = pq.WilsonParams(mass=0.1)

    # Apply with antiperiodic BC
    bc_anti = pq.BoundaryCondition(fermion_bc_time=-1.0)
    lat.build_neighbor_tables(bc_anti)
    result_anti = pq.apply_M(U, psi, params, bc_anti)

    # Apply with periodic BC
    bc_periodic = pq.BoundaryCondition(fermion_bc_time=1.0)
    lat.build_neighbor_tables(bc_periodic)
    result_periodic = pq.apply_M(U, psi, params, bc_periodic)

    # Results should be different
    diff = torch.abs(result_anti.site - result_periodic.site).max().item()
    assert diff > 1e-10, "BC change had no effect on result"


def test_gauge_field_identity() -> None:
    """Test that identity gauge field is correct."""
    lat = pq.Lattice((4, 4, 4, 4))
    U = pq.GaugeField.eye(lat)

    eye = torch.eye(3, dtype=U.dtype, device=U.device)

    for mu in range(4):
        for site in range(lat.volume):
            diff = torch.abs(U[mu][site] - eye).max().item()
            assert diff < 1e-15, f"Identity gauge not correct at mu={mu}, site={site}"


def test_lattice_neighbor_tables() -> None:
    """Test that neighbor tables are correctly computed."""
    lat = pq.Lattice((4, 4, 4, 4))
    bc = pq.BoundaryCondition()
    lat.build_neighbor_tables(bc)

    # Check that forward + backward gives back original for bulk sites
    # and applies BC phases at boundaries
    for mu in range(4):
        fwd_idx, fwd_phase = lat.neighbor_fwd(mu)
        _bwd_idx, _bwd_phase = lat.neighbor_bwd(mu)

        for site in range(lat.volume):
            x, y, z, t = lat.coord(site)

            # Forward neighbor
            coords_fwd = [x, y, z, t]
            expected_fwd_coord = (coords_fwd[mu] + 1) % lat.shape[mu]
            coords_fwd[mu] = expected_fwd_coord
            expected_fwd = lat.index(*coords_fwd)

            assert fwd_idx[site].item() == expected_fwd, (
                f"Forward neighbor wrong at site {site}, mu={mu}"
            )

            # Check BC phase
            original_coord = [x, y, z, t][mu]
            expected_phase = bc.get_bc_phase(mu) if original_coord == lat.shape[mu] - 1 else 1.0
            assert abs(fwd_phase[site].real.item() - expected_phase) < 1e-15
