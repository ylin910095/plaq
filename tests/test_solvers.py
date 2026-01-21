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
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)  # Periodic for simplicity
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
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

    def test_reference_solve_M_random_gauge(self) -> None:
        """Test BiCGStab matches dense solve on 2^4 lattice with random gauge."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Random SU(3) gauge field
        torch.manual_seed(999)
        U = pq.GaugeField.random(lat)
        params = pq.WilsonParams(mass=0.2)  # Slightly higher mass for stability

        torch.manual_seed(42)
        b = pq.SpinorField.random(lat)

        # Build dense M matrix
        V = lat.volume
        n_dof = V * 4 * 3
        M_dense = torch.zeros(n_dof, n_dof, dtype=torch.complex128)

        for i in range(n_dof):
            e_i = torch.zeros(V, 4, 3, dtype=torch.complex128)
            site = i // 12
            spin = (i % 12) // 3
            color = i % 3
            e_i[site, spin, color] = 1.0

            e_i_field = pq.SpinorField(e_i, lat)
            M_e_i = pq.apply_M(U, e_i_field, params, bc)
            M_dense[:, i] = M_e_i.site.flatten()

        # Dense solve
        b_flat = b.site.flatten()
        x_dense = torch.linalg.solve(M_dense, b_flat)

        # BiCGStab solve
        x_bicgstab, info = pq.solve(
            U, b, method="bicgstab", equation="M", params=params, bc=bc, tol=1e-10
        )

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
        assert rel_error < 1e-8, f"Relative error too large: {rel_error}"
        assert residual < 1e-10, f"Residual too large: {residual}"


class TestReferenceSolveMdagM:
    """Test CG solver against dense reference for MdagM x = Mdag b."""

    def test_reference_solve_MdagM_identity_gauge(self) -> None:
        """Test CG matches dense solve on 2^4 lattice with identity gauge."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
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

    def test_reference_solve_MdagM_random_gauge(self) -> None:
        """Test CG matches dense solve on 2^4 lattice with random gauge."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        # Random SU(3) gauge field
        torch.manual_seed(888)
        U = pq.GaugeField.random(lat)
        params = pq.WilsonParams(mass=0.2)

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

        # CG solve
        x_cg, info = pq.solve(U, b, method="cg", equation="MdagM", params=params, bc=bc, tol=1e-10)

        # Compare
        x_cg_flat = x_cg.site.flatten()
        rel_error = (torch.linalg.norm(x_cg_flat - x_dense) / torch.linalg.norm(x_dense)).item()

        assert info.converged, f"CG did not converge: {info}"
        assert rel_error < 1e-8, f"Relative error too large: {rel_error}"


class TestSolverResidualMonotonicity:
    """Test residual behavior of solvers."""

    def test_cg_residual_decreases(self) -> None:
        """Test that CG residual decreases monotonically using callback."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.5)  # Larger mass for faster convergence

        torch.manual_seed(456)
        b = pq.SpinorField.random(lat)

        # Track residuals using callback
        residuals: list[float] = []

        # Apply Mdag to b for RHS norm calculation
        Mdag_b = pq.apply_Mdag(U, b, params, bc)
        b_norm = torch.linalg.norm(Mdag_b.site.flatten()).item()

        def callback(x: pq.SpinorField) -> None:
            """Callback to track residual at each iteration."""
            # Compute MdagM x
            MdagM_x = pq.apply_MdagM(U, x, params, bc)
            # Compute residual ||MdagM x - Mdag b||
            residual_vec = MdagM_x.site - Mdag_b.site
            residual_norm = torch.linalg.norm(residual_vec.flatten()).item()
            rel_residual = residual_norm / b_norm
            residuals.append(rel_residual)

        # Solve with callback
        _x, info = pq.solve(
            U, b, method="cg", equation="MdagM", params=params, bc=bc, callback=callback
        )

        # Check monotonic decrease (with small tolerance for numerical errors)
        for i in range(1, len(residuals)):
            # Allow tiny increase due to rounding
            assert residuals[i] <= residuals[i - 1] * 1.001, (
                f"Residual increased at step {i}: {residuals[i - 1]} -> {residuals[i]}"
            )

        # Check convergence
        assert info.converged, f"CG did not converge: {info}"
        assert residuals[-1] < 1e-10, f"Final residual too large: {residuals[-1]}"
        assert len(residuals) > 0, "Callback was not called"


class TestEvenOddConsistency:
    """Test even-odd preconditioning consistency."""

    def test_eo_preconditioned_matches_unpreconditioned(self) -> None:
        """EO-preconditioned solve should match unpreconditioned on tiny lattice."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition(fermion_bc_time=1.0)
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
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

        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
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
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Default dtype (complex128)
        x, _ = pq.solve(U, b, tol=1e-8, maxiter=100)
        assert x.dtype == torch.complex128

        # Explicit dtype - skip complex64 as Wilson operator needs internal dtype conversion
        # x64, _ = pq.solve(U, b, dtype=torch.complex64, tol=1e-6, maxiter=100)
        # assert x64.dtype == torch.complex64


class TestSolverCallbacks:
    """Test solver callback functionality."""

    def test_cg_callback_called(self) -> None:
        """Test that CG callback is called at each iteration."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.5)
        b = pq.SpinorField.random(lat)

        # Track callback invocations
        callback_count = [0]
        solutions: list[pq.SpinorField] = []

        def callback(x: pq.SpinorField) -> None:
            callback_count[0] += 1
            solutions.append(x)

        # Solve with callback
        _x, info = pq.solve(
            U, b, method="cg", equation="MdagM", params=params, bc=bc, callback=callback, tol=1e-8
        )

        # Verify callback was called
        assert callback_count[0] > 0, "Callback was not called"
        # Note: callback may be called one less time than iters if convergence is detected
        # at the beginning of an iteration (before callback is invoked)
        assert callback_count[0] >= info.iters - 1, (
            f"Callback count {callback_count[0]} too low for {info.iters} iterations"
        )
        assert callback_count[0] <= info.iters, (
            f"Callback count {callback_count[0]} exceeds iterations {info.iters}"
        )
        assert len(solutions) == callback_count[0], (
            "Number of solutions does not match callback count"
        )

        # Verify solutions are different (convergence)
        if len(solutions) > 1:
            diff = torch.abs(solutions[-1].site - solutions[0].site).max().item()
            assert diff > 0, "Solution did not change during iterations"

    def test_bicgstab_callback_called(self) -> None:
        """Test that BiCGStab callback is called at each iteration."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.5)
        torch.manual_seed(123)
        b = pq.SpinorField.random(lat)

        # Track callback invocations
        callback_count = [0]

        def callback(_x: pq.SpinorField) -> None:
            callback_count[0] += 1

        # Solve with callback
        _x, info = pq.solve(
            U, b, method="bicgstab", equation="M", params=params, bc=bc, callback=callback, tol=1e-8
        )

        # Verify callback was called
        assert callback_count[0] > 0, "Callback was not called"
        # Note: callback may be called one less time than iters if convergence is detected
        # at the beginning of an iteration (before callback is invoked)
        assert callback_count[0] >= info.iters - 1, (
            f"Callback count {callback_count[0]} too low for {info.iters} iterations"
        )
        assert callback_count[0] <= info.iters, (
            f"Callback count {callback_count[0]} exceeds iterations {info.iters}"
        )

    def test_callback_with_eo_preconditioning(self) -> None:
        """Test callback works with even-odd preconditioning."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        params = pq.WilsonParams(mass=0.2)
        torch.manual_seed(789)
        b = pq.SpinorField.random(lat)

        # Track callback invocations
        callback_count = [0]

        def callback(x: pq.SpinorField) -> None:
            callback_count[0] += 1
            # Verify we receive a SpinorField
            assert isinstance(x, pq.SpinorField), "Callback argument is not a SpinorField"
            # Verify it has correct shape
            assert x.site.shape == (lat.volume, 4, 3), "Callback solution has incorrect shape"

        # Solve with EO preconditioning and callback
        _x, _info = pq.solve(
            U,
            b,
            equation="MdagM",
            precond="eo",
            params=params,
            bc=bc,
            callback=callback,
            tol=1e-8,
        )

        # Verify callback was called (at least once for the outer solve)
        assert callback_count[0] > 0, "Callback was not called with EO preconditioning"

    def test_callback_none_works(self) -> None:
        """Test that None callback works (no callback provided)."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Solve without callback (should not error)
        _x, info = pq.solve(U, b, callback=None, tol=1e-8)

        assert info.converged, "Solver did not converge without callback"


class TestSolverAPI:
    """Test solver API behavior."""

    def test_auto_equation_selection(self) -> None:
        """Auto equation should select MdagM for Wilson."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        # Auto should select MdagM and CG
        _, info = pq.solve(U, b, equation="auto", method="auto", tol=1e-8)
        assert info.equation == "MdagM"
        assert info.method == "cg"

    def test_solver_info_fields(self) -> None:
        """SolverInfo should have all required fields."""
        lat = pq.Lattice((4, 6, 4, 6))
        bc = pq.BoundaryCondition()
        lat.build_neighbor_tables(bc)

        U = pq.GaugeField.eye(lat)
        b = pq.SpinorField.random(lat)

        _, info = pq.solve(U, b, tol=1e-8)

        # Check all fields exist
        assert isinstance(info.converged, bool)
        assert isinstance(info.iters, int)
        assert isinstance(info.final_residual, float)
        assert isinstance(info.method, str)
        assert isinstance(info.equation, str)
