"""
Setup script for quda_torch_op PyTorch extension.
Builds CPU-only extension using PyTorch's stable C++ API.
"""

from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Check PyTorch version for stable ABI support
py_limited_api = torch.__version__ >= "2.6.0"

# Compile arguments for C++
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
extra_link_args = []

extra_link_args.extend(
    [
        f"-Wl,-rpath,{torch_lib_path}",
    ]
)

# Define the C++ extension
ext_modules = [
    CppExtension(
        name="quda_torch_op._C",
        sources=["csrc/code.cpp"],
        extra_compile_args={"cxx": cxx_args},
        extra_link_args=extra_link_args,
        py_limited_api=py_limited_api,
    ),
]

# Wheel options for stable ABI
options = {}
if py_limited_api:
    options["bdist_wheel"] = {"py_limited_api": "cp311"}

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options=options,
)
