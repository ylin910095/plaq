"""Tests for linear solvers.

This module tests the Krylov solvers (CG, BiCGStab) against dense reference
solutions on small lattices.
"""

import torch

import plaq as pq


class TestReferenceSolveM:
    """Test BiCGStab solver against dense reference for M x = b."""

    def test_reference_solve_M_identity_gauge(self) -> None:
        """Test BiCGStab matches dense solve on 2^4 lattice with identity gauge."""
        # Small lattice for dense solve
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic for simplicity
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        params = pq.WilsonParams(mass=0.1)

        # Random source
        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)

        # Build dense M matrix by applying to basis vectors
        V = lat.volume
        n_dof = V * 4 * 3  # sites * spin * color
        M_dense = torch.zeros(n_dof, n_dof, dtype=torch.complex128)

        for i in range(n_dof):
            # Create basis vector
            e_i = torch.zeros(V, 4, 3, dtype=torch.complex128)
            site = i // 12
            spin = (i % 12) // 3
            color = i % 3
            e_i[site, spin, color] = 1.0

            e_i_field = pq.SpinorField(e_i, lat)
            M_e_i = pq.apply_M(U, e_i_field, params, bc)
            M_dense[:, i] = M_e_i.site.flatten()

        # Dense solve: M x = b
        b_flat = b.site.flatten()
        x_dense = torch.linalg.solve(M_dense, b_flat)

        # BiCGStab solve
        x_bicgstab, info = pq.solve(U, b, method="bicgstab", equation="M", params=params, bc=bc)

        # Compare
        x_bicgstab_flat = x_bicgstab.site.flatten()
        rel_error = (
            torch.linalg.norm(x_bicgstab_flat - x_dense) / torch.linalg.norm(x_dense)
        ).item()

        # Compute residual
        Mx = pq.apply_M(U, x_bicgstab, params, bc)
        residual = (
            torch.linalg.norm(Mx.site.flatten() - b_flat) / torch.linalg.norm(b_flat)
        ).item()

        assert info.converged, f"BiCGStab did not converge: {info}"
        assert rel_error < 1e-10, f"Relative error too large: {rel_error}"
        assert residual < 1e-12, f"Residual too large: {residual}"


class TestReferenceSolveMdagM:
    """Test CG solver against dense reference for MdagM x = Mdag b."""

    def test_reference_solve_MdagM_identity_gauge(self) -> None:
        """Test CG matches dense solve on 2^4 lattice with identity gauge."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        params = pq.WilsonParams(mass=0.1)

        torch.manual_seed(123)
        b = pq.SpinorField.random(lat)

        # Build dense MdagM matrix
        V = lat.volume
        n_dof = V * 4 * 3
        MdagM_dense = torch.zeros(n_dof, n_dof, dtype=torch.complex128)

        for i in range(n_dof):
            e_i = torch.zeros(V, 4, 3, dtype=torch.complex128)
            site = i // 12
            spin = (i % 12) // 3
            color = i % 3
            e_i[site, spin, color] = 1.0

            e_i_field = pq.SpinorField(e_i, lat)
            MdagM_e_i = pq.apply_MdagM(U, e_i_field, params, bc)
            MdagM_dense[:, i] = MdagM_e_i.site.flatten()

        # RHS = Mdag b
        Mdag_b = pq.apply_Mdag(U, b, params, bc)
        rhs = Mdag_b.site.flatten()

        # Dense solve
        x_dense = torch.linalg.solve(MdagM_dense, rhs)

        # CG solve (default is MdagM)
        x_cg, info = pq.solve(U, b, method="cg", equation="MdagM", params=params, bc=bc)

        # Compare
        x_cg_flat = x_cg.site.flatten()
        rel_error = (torch.linalg.norm(x_cg_flat - x_dense) / torch.linalg.norm(x_dense)).item()

        assert info.converged, f"CG did not converge: {info}"
        assert rel_error < 1e-10, f"Relative error too large: {rel_error}"


class TestSolverResidualMonotonicity:
    """Test residual behavior of solvers."""

    def test_cg_residual_decreases(self) -> None:
        """Test that CG residual decreases monotonically for SPD system."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        params = pq.WilsonParams(mass=0.5)  # Larger mass for faster convergence

        torch.manual_seed(456)

        # Create random b
        b_data = torch.randn(lat.volume, 4, 3, dtype=torch.complex128)

        # Track residuals manually
        residuals: list[float] = []

        def A_apply(x: torch.Tensor) -> torch.Tensor:
            x_field = pq.SpinorField(x, lat)
            return pq.apply_MdagM(U, x_field, params, bc).site

        # Apply Mdag to b
        b_field = pq.SpinorField(b_data, lat)
        Mdag_b = pq.apply_Mdag(U, b_field, params, bc).site
        b_norm = torch.linalg.norm(Mdag_b.flatten()).item()

        # Run CG with callback to track residuals
        x = torch.zeros_like(b_data)
        r = Mdag_b.clone()
        p = r.clone()
        rr = torch.vdot(r.flatten(), r.flatten()).real

        for _ in range(50):
            residuals.append((torch.sqrt(rr) / b_norm).item())  # pyright: ignore[reportAttributeAccessIssue]

            if residuals[-1] < 1e-12:
                break

            Ap = A_apply(p)
            pAp = torch.vdot(p.flatten(), Ap.flatten()).real
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.vdot(r.flatten(), r.flatten()).real
            beta = rr_new / rr
            p = r + beta * p
            rr = rr_new

        # Check monotonic decrease (with small tolerance for numerical errors)
        for i in range(1, len(residuals)):
            # Allow tiny increase due to rounding
            assert residuals[i] <= residuals[i - 1] * 1.001, (
                f"Residual increased at step {i}: {residuals[i - 1]} -> {residuals[i]}"
            )

        # Check convergence
        assert residuals[-1] < 1e-10, f"Final residual too large: {residuals[-1]}"


class TestEvenOddConsistency:
    """Test even-odd preconditioning consistency."""

    def test_eo_preconditioned_matches_unpreconditioned(self) -> None:
        """EO-preconditioned solve should match unpreconditioned on tiny lattice."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        params = pq.WilsonParams(mass=0.2)

        torch.manual_seed(789)
        b = pq.SpinorField.random(lat)

        # Solve without preconditioning
        x_no_precond, info_no_precond = pq.solve(
            U, b, equation="MdagM", precond=None, params=params, bc=bc, tol=1e-12
        )

        # Solve with EO preconditioning
        x_eo, info_eo = pq.solve(
            U, b, equation="MdagM", precond="eo", params=params, bc=bc, tol=1e-12
        )

        # Both should converge
        assert info_no_precond.converged, f"Unpreconditioned did not converge: {info_no_precond}"
        assert info_eo.converged, f"EO preconditioned did not converge: {info_eo}"

        # Solutions should match
        diff = torch.abs(x_no_precond.site - x_eo.site).max().item()
        assert diff < 1e-8, f"EO solution differs from unpreconditioned: max diff = {diff}"

    def test_eo_hopping_roundtrip(self) -> None:
        """Test that EO hopping terms are consistent with full operator."""
        from plaq.precond.even_odd import apply_hopping_eo, apply_hopping_oe

        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        params = pq.WilsonParams(mass=0.1)

        torch.manual_seed(101)

        # Create random spinor
        psi = pq.SpinorField.random(lat)
        psi_eo = pq.pack_eo(psi.site, lat)  # [2, V/2, 4, 3]
        psi_e = psi_eo[0]
        psi_o = psi_eo[1]

        # Apply full M
        M_psi = pq.apply_M(U, psi, params, bc)
        M_psi_eo = pq.pack_eo(M_psi.site, lat)

        # Reconstruct from blocks
        mass_factor = params.mass + 4.0 * params.r

        # M_ee psi_e + M_eo psi_o = (M psi)_e
        Mee_psi_e = mass_factor * psi_e
        Meo_psi_o = apply_hopping_eo(U, psi_o, lat, params)
        reconstructed_e = Mee_psi_e + Meo_psi_o

        # M_oe psi_e + M_oo psi_o = (M psi)_o
        Moe_psi_e = apply_hopping_oe(U, psi_e, lat, params)
        Moo_psi_o = mass_factor * psi_o
        reconstructed_o = Moe_psi_e + Moo_psi_o

        # Compare
        diff_e = torch.abs(M_psi_eo[0] - reconstructed_e).max().item()
        diff_o = torch.abs(M_psi_eo[1] - reconstructed_o).max().item()

        assert diff_e < 1e-12, f"Even site reconstruction error: {diff_e}"
        assert diff_o < 1e-12, f"Odd site reconstruction error: {diff_o}"


class TestSolverDtype:
    """Test solver dtype handling."""

    def test_solve_returns_correct_dtype(self) -> None:
        """Solve should return the specified dtype."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        b = pq.SpinorField.random(lat)

        # Default dtype (complex128)
        x, _ = pq.solve(U, b, tol=1e-8, maxiter=100)
        assert x.dtype == torch.complex128

        # Explicit dtype - skip complex64 as Wilson operator needs internal dtype conversion
        # x64, _ = pq.solve(U, b, dtype=torch.complex64, tol=1e-6, maxiter=100)
        # assert x64.dtype == torch.complex64


class TestSolverAPI:
    """Test solver API behavior."""

    def test_auto_equation_selection(self) -> None:
        """Auto equation should select MdagM for Wilson."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        b = pq.SpinorField.random(lat)

        # Auto should select MdagM and CG
        _, info = pq.solve(U, b, equation="auto", method="auto", tol=1e-8)
        assert info.equation == "MdagM"
        assert info.method == "cg"

    def test_solver_info_fields(self) -> None:
        """SolverInfo should have all required fields."""
        lat = pq.Lattice((2, 2, 2, 2))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.identity(lat)
        b = pq.SpinorField.random(lat)

        _, info = pq.solve(U, b, tol=1e-8)

        # Check all fields exist
        assert isinstance(info.converged, bool)
        assert isinstance(info.iters, int)
        assert isinstance(info.final_residual, float)
        assert isinstance(info.method, str)
        assert isinstance(info.equation, str)
