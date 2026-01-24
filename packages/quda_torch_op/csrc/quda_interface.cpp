/**
 * QUDA interface for quda_torch_op
 *
 * This file provides PyTorch operator implementations that interface with QUDA.
 * The operator schema is defined in code.cpp; this file only registers implementations.
 *
 * Note: QUDA is built with MPI support, so we initialize MPI for single-process use.
 */

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <quda.h>
#include <device.h>
#include <mpi.h>

#include <atomic>
#include <mutex>
#include <string>

namespace quda_torch_op {

// Thread-safe QUDA initialization state
static std::atomic<bool> g_quda_initialized{false};
static std::mutex g_quda_mutex;
static int g_quda_device = -1;

/**
 * Check if QUDA is available (compiled with QUDA support).
 */
bool quda_is_available() {
    return true;  // If this code is compiled, QUDA is available
}

/**
 * Get number of CUDA devices.
 */
int64_t quda_get_device_count() {
    return quda::device::get_device_count();
}

/**
 * Initialize QUDA on a specific device.
 *
 * Args:
 *     device: CUDA device index (0-based). Use -1 for default device.
 *
 * This function is idempotent - calling it multiple times with the same
 * device is safe. However, calling with a different device after initialization
 * will raise an error.
 */
void quda_init(int64_t device) {
    std::lock_guard<std::mutex> lock(g_quda_mutex);

    if (g_quda_initialized.load()) {
        // Already initialized - check if same device
        if (g_quda_device != device && device != -1) {
            STD_TORCH_CHECK(
                false,
                "QUDA already initialized on device " + std::to_string(g_quda_device) +
                ", cannot reinitialize on device " + std::to_string(device));
        }
        return;
    }

    int num_devices = quda::device::get_device_count();
    STD_TORCH_CHECK(
        num_devices > 0,
        "No CUDA devices found. QUDA requires at least one GPU.");

    int target_device = (device == -1) ? 0 : static_cast<int>(device);
    STD_TORCH_CHECK(
        target_device >= 0 && target_device < num_devices,
        "Invalid device index " + std::to_string(target_device) +
        ". Available devices: 0-" + std::to_string(num_devices - 1));

    // Initialize MPI if not already initialized (QUDA requires MPI)
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    // Set up single-process communication grid (1x1x1x1)
    int comm_dims[4] = {1, 1, 1, 1};
    initCommsGridQuda(4, comm_dims, nullptr, nullptr);

    // Set verbosity to silent for library use
    setVerbosityQuda(QUDA_SILENT, "", stdout);

    // Initialize QUDA
    initQuda(target_device);

    g_quda_device = target_device;
    g_quda_initialized.store(true);
}

/**
 * Finalize QUDA and release resources.
 *
 * This should be called when QUDA is no longer needed to free GPU memory.
 * Note: MPI is NOT finalized because it's a process-wide resource that
 * cannot be restarted once finalized. MPI will be cleaned up when the
 * process exits.
 */
void quda_finalize() {
    std::lock_guard<std::mutex> lock(g_quda_mutex);

    if (!g_quda_initialized.load()) {
        return;  // Nothing to do
    }

    endQuda();

    // Note: We intentionally do NOT call MPI_Finalize() here.
    // MPI cannot be restarted once finalized, and the library should
    // support reinitializing QUDA in long-running processes.
    // MPI will be cleaned up automatically when the process exits.

    g_quda_initialized.store(false);
    g_quda_device = -1;
}

/**
 * Check if QUDA has been initialized.
 */
bool quda_is_initialized() {
    return g_quda_initialized.load();
}

/**
 * Get the device QUDA was initialized on.
 * Returns -1 if not initialized.
 */
int64_t quda_get_device() {
    return g_quda_device;
}

/**
 * Get QUDA version string.
 */
std::string quda_get_version() {
    // QUDA doesn't have a runtime version function, use compile-time version
    return std::string("1.1.0");
}

// Register QUDA implementations
// Note: The operator schema is defined in code.cpp
STABLE_TORCH_LIBRARY_IMPL(quda_torch_op, CompositeExplicitAutograd, m) {
    m.impl("quda_is_available", TORCH_BOX(&quda_is_available));
    m.impl("quda_get_device_count", TORCH_BOX(&quda_get_device_count));
    m.impl("quda_init", TORCH_BOX(&quda_init));
    m.impl("quda_finalize", TORCH_BOX(&quda_finalize));
    m.impl("quda_is_initialized", TORCH_BOX(&quda_is_initialized));
    m.impl("quda_get_device", TORCH_BOX(&quda_get_device));
    m.impl("quda_get_version", TORCH_BOX(&quda_get_version));
}

}  // namespace quda_torch_op
