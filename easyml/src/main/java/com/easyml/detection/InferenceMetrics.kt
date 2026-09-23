package com.easyml.detection

/**
 * Granular latency metrics for the object detection pipeline measured via high-precision [System.nanoTime].
 *
 * Separating preprocessing, TFLite execution, and postprocessing enables accurate profiling
 * and isolates hardware acceleration performance from CPU bitmap manipulation overhead.
 *
 * @param preprocessMs Duration of letterboxing, pixel extraction, and normalization in milliseconds
 * @param inferenceMs Duration of TFLite tensor execution on the hardware delegate in milliseconds
 * @param postprocessMs Duration of output tensor decoding, coordinate unmapping, and NMS in milliseconds
 * @param totalMs Total pipeline latency from input Bitmap to final Detection list in milliseconds
 */
data class InferenceMetrics(
    val preprocessMs: Double = 0.0,
    val inferenceMs: Double = 0.0,
    val postprocessMs: Double = 0.0,
    val totalMs: Double = 0.0
) {
    /** Estimated frames per second based on total pipeline latency */
    val fps: Double
        get() = if (totalMs > 0.0) 1000.0 / totalMs else 0.0

    override fun toString(): String =
        "InferenceMetrics(prep=%.1fms, inf=%.1fms, post=%.1fms, total=%.1fms, fps=%.1f)"
            .format(preprocessMs, inferenceMs, postprocessMs, totalMs, fps)
}
