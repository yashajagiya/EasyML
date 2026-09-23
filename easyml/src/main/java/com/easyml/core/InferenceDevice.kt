package com.easyml.core

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

/**
 * Hardware acceleration target for inference.
 *
 * - [AUTO]: Recommended default. Automatically checks device GPU compatibility;
 *          uses GPU delegate if supported, or seamlessly falls back to CPU
 *          with XNNPACK (ARM NEON SIMD) acceleration.
 * - [GPU]: Uses the device's GPU via OpenGL/OpenCL (2-5x faster for neural networks).
 * - [CPU]: Fast multi-threaded CPU inference powered by XNNPACK.
 * - [NNAPI]: Android Neural Networks API.
 */
enum class InferenceDevice {
    AUTO,
    GPU,
    CPU,
    NNAPI;

    /**
     * Applies the appropriate delegate configuration to TFLite Interpreter.Options.
     * Returns a GpuDelegate if GPU is selected (caller must close it), or null.
     */
    internal fun configure(options: Interpreter.Options, numThreads: Int = 4): GpuDelegate? {
        // Optimal thread count clamped to available cores to prevent core contention and overheating
        val availableCores = Runtime.getRuntime().availableProcessors()
        val optimalThreads = minOf(availableCores, numThreads.coerceIn(1, 4))
        options.setNumThreads(optimalThreads)

        return when (this) {
            AUTO -> {
                val gpu = createGpuDelegate(forceGpu = false)
                if (gpu != null) {
                    options.addDelegate(gpu)
                    gpu
                } else {
                    // Fall back to CPU with XNNPACK SIMD acceleration
                    options.setUseXNNPACK(true)
                    null
                }
            }
            GPU -> {
                val gpu = createGpuDelegate(forceGpu = true)
                if (gpu != null) {
                    options.addDelegate(gpu)
                    gpu
                } else {
                    Log.w("EasyML", "GPU delegate unavailable, falling back to CPU with XNNPACK")
                    options.setUseXNNPACK(true)
                    null
                }
            }
            CPU -> {
                options.setUseXNNPACK(true)
                null
            }
            NNAPI -> {
                try {
                    options.setUseNNAPI(true)
                } catch (e: Exception) {
                    Log.w("EasyML", "NNAPI not available, using XNNPACK CPU: ${e.message}")
                    options.setUseXNNPACK(true)
                }
                null
            }
        }
    }

    private fun createGpuDelegate(forceGpu: Boolean): GpuDelegate? {
        return try {
            val isSupported = try {
                CompatibilityList().isDelegateSupportedOnThisDevice
            } catch (_: Throwable) {
                false
            }

            if (isSupported || forceGpu) {
                GpuDelegate()
            } else {
                null
            }
        } catch (e: Throwable) {
            Log.w("EasyML", "GPU initialization failed, using CPU: ${e.message}")
            null
        }
    }
}
