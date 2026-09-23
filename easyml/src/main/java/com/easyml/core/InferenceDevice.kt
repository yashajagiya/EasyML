@file:Suppress("SpellCheckingInspection")
package com.easyml.core

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory

/**
 * Hardware acceleration target for model inference.
 *
 * - [AUTO]: Recommended default. Tries GPU delegate with FP16 precision, then NNAPI,
 *          then seamlessly falls back to CPU with XNNPACK SIMD acceleration.
 * - [GPU]: Attempts GPU inference with FP16 precision. Falls back to CPU if unsupported.
 * - [GPU_STRICT]: Strict GPU execution. Throws [IllegalStateException] if GPU delegate fails to initialize.
 * - [CPU]: Multi-threaded CPU inference powered by XNNPACK (ARM NEON / AVX).
 * - [NNAPI]: Android Neural Networks API hardware acceleration.
 */
enum class InferenceDevice {
    AUTO,
    GPU,
    GPU_STRICT,
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
            GPU_STRICT -> {
                val gpu = createGpuDelegate(useFp16 = useFp16)
                    ?: throw IllegalStateException("GPU acceleration was strictly requested (InferenceDevice.GPU_STRICT), but GPU delegate initialization failed on this device.")
                options.addDelegate(gpu)
                gpu
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
            val delegateOptions = GpuDelegateFactory.Options().apply {
                isPrecisionLossAllowed = useFp16
                inferencePreference = GpuDelegateFactory.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
            }
            GpuDelegate(delegateOptions)
        } catch (e: Throwable) {
            Log.w("EasyML", "GPU initialization failed: ${e.message}")
            null
        }
    }
}
