"""
Setup script for quda_torch_op PyTorch extension.

Builds extension with optional QUDA support. QUDA linking is enabled when:
1. QUDA_HOME environment variable is set to the QUDA installation directory
2. MPI_HOME environment variable is set to the MPI installation directory
3. A CUDA-capable GPU is available

Usage:
    # Without QUDA (CPU-only ops):
    pip install .

    # With QUDA:
    export QUDA_HOME=/opt/quda/install
    export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
    pip install .
"""

import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_cuda_device_count() -> int:
    """Get number of CUDA devices, returns 0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def find_quda_home() -> Path | None:
    """Find QUDA installation directory from QUDA_HOME env var."""
    quda_home = os.environ.get("QUDA_HOME")
    if quda_home is None:
        return None

    quda_path = Path(quda_home)
    if not quda_path.exists():
        print(f"Warning: QUDA_HOME={quda_home} does not exist", file=sys.stderr)
        return None

    return quda_path


def find_mpi_home() -> Path | None:
    """Find MPI installation directory from MPI_HOME env var."""
    mpi_home = os.environ.get("MPI_HOME")
    if mpi_home is None:
        return None

    mpi_path = Path(mpi_home)
    if not mpi_path.exists():
        print(f"Warning: MPI_HOME={mpi_home} does not exist", file=sys.stderr)
        return None

    return mpi_path


def validate_quda_installation(quda_home: Path) -> bool:
    """Validate that QUDA installation has required files."""
    lib_path = quda_home / "lib" / "libquda.so"
    include_path = quda_home / "include" / "quda.h"

    if not lib_path.exists():
        print(f"Error: QUDA library not found at {lib_path}", file=sys.stderr)
        print(
            "Please ensure QUDA is properly installed and QUDA_HOME points to the install directory.",
            file=sys.stderr,
        )
        return False

    if not include_path.exists():
        print(f"Error: QUDA headers not found at {include_path}", file=sys.stderr)
        print(
            "Please ensure QUDA is properly installed and QUDA_HOME points to the install directory.",
            file=sys.stderr,
        )
        return False

    return True


def check_quda_availability() -> tuple[bool, str]:
    """
    Check if QUDA can be used.

    Returns:
        Tuple of (is_available, message)
    """
    # Check for CUDA
    cuda_count = get_cuda_device_count()
    if cuda_count == 0:
        return False, "No CUDA-capable GPU found. QUDA requires a GPU."

    # Check for QUDA_HOME
    quda_home = find_quda_home()
    if quda_home is None:
        return (
            False,
            "QUDA_HOME environment variable not set. "
            "Set QUDA_HOME to your QUDA installation directory to enable QUDA support.",
        )

    # Validate installation
    if not validate_quda_installation(quda_home):
        return False, f"QUDA installation at {quda_home} is incomplete."

    # Check for MPI_HOME
    mpi_home = find_mpi_home()
    if mpi_home is None:
        return (
            False,
            "MPI_HOME environment variable not set. "
            "Set MPI_HOME to your MPI installation directory (e.g., /usr/lib/x86_64-linux-gnu/openmpi).",
        )

    return True, f"QUDA found at {quda_home} with {cuda_count} GPU(s) available."


# Check PyTorch version for stable ABI support
py_limited_api = torch.__version__ >= "2.6.0"

# Base compile arguments for C++
cxx_args = ["-O3", "-std=c++17"]

# Add stable ABI defines if supported
if py_limited_api:
    cxx_args.extend(
        [
            "-DPy_LIMITED_API=0x030b0000",  # Python 3.11+ limited API
            "-DTORCH_TARGET_VERSION=0x020a000000000000",  # PyTorch 2.10.0 min
        ]
    )

# Link arguments - set rpath to find PyTorch libraries at runtime
torch_lib_path = Path(torch.__file__).parent / "lib"
extra_link_args = [f"-Wl,-rpath,{torch_lib_path}"]

# Determine QUDA availability
quda_available, quda_message = check_quda_availability()
print(f"QUDA status: {quda_message}")

# Source files (always include base code)
sources = ["csrc/code.cpp"]

# Determine extension type and configure QUDA if available
if quda_available:
    quda_home = find_quda_home()
    quda_include = quda_home / "include"
    quda_lib = quda_home / "lib"

    # Add QUDA-specific sources
    sources.append("csrc/quda_interface.cpp")
    sources.append("csrc/wilson_interface.cpp")

    # Add QUDA and MPI include and library paths
    # QUDA was built with MPI support, so we need MPI headers and libs
    mpi_home = find_mpi_home()
    mpi_include = mpi_home / "include"
    mpi_lib = mpi_home / "lib"

    include_dirs = [str(quda_include), str(mpi_include)]
    library_dirs = [str(quda_lib), str(mpi_lib)]
    libraries = ["quda", "mpi"]

    # Add QUDA compile flag
    cxx_args.append("-DQUDA_ENABLED=1")

    # Add QUDA rpath
    extra_link_args.append(f"-Wl,-rpath,{quda_lib}")
    extra_link_args.append(f"-Wl,-rpath,{mpi_lib}")

    # Use CppExtension with QUDA linking (QUDA handles CUDA internally)
    ext_modules = [
        CppExtension(
            name="quda_torch_op._C",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={"cxx": cxx_args},
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        ),
    ]
    print("Building with QUDA support enabled.")
else:
    # CPU-only build
    cxx_args.append("-DQUDA_ENABLED=0")

    ext_modules = [
        CppExtension(
            name="quda_torch_op._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_args},
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        ),
    ]
    print("Building without QUDA support (CPU-only).")

# Wheel options for stable ABI
options = {}
if py_limited_api:
    options["bdist_wheel"] = {"py_limited_api": "cp311"}

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options=options,
)
