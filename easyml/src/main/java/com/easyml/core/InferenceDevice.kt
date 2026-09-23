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
    internal fun configure(
        options: Interpreter.Options,
        numThreads: Int = 4,
        useFp16: Boolean = true
    ): GpuDelegate? {
        val availableCores = Runtime.getRuntime().availableProcessors()
        val optimalThreads = minOf(availableCores, numThreads.coerceIn(1, 4))
        options.setNumThreads(optimalThreads)

        return when (this) {
            AUTO -> {
                val gpu = createGpuDelegate(useFp16 = useFp16)
                if (gpu != null) {
                    options.addDelegate(gpu)
                    gpu
                } else {
                    // Try NNAPI or fall back to CPU with XNNPACK
                    try {
                        options.setUseNNAPI(true)
                        null
                    } catch (_: Throwable) {
                        options.setUseXNNPACK(true)
                        null
                    }
                }
            }
            GPU -> {
                val gpu = createGpuDelegate(useFp16 = useFp16)
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

    private fun createGpuDelegate(useFp16: Boolean): GpuDelegate? {
        return try {
            val delegateOptions = GpuDelegate.Options().apply {
                setPrecisionLossAllowed(useFp16)
                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER)
            }
            GpuDelegate(delegateOptions)
        } catch (e: Throwable) {
            Log.w("EasyML", "GPU initialization failed: ${e.message}")
            null
        }
    }
}
