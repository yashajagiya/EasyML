package com.easyml.detection

import com.easyml.core.InferenceDevice
import com.easyml.core.LabelSource
import com.easyml.core.ModelSource

/**
 * DSL configuration for [ObjectDetector].
 *
 * Usage:
 * ```kotlin
 * EasyML.objectDetector(context) {
 *     model = ModelSource.Asset("yolon.tflite")
 *     labels = LabelSource.Asset("labels.txt")
 *     confidenceThreshold = 0.5f
 *     iouThreshold = 0.45f
 *     maxResults = 20
 *     device = InferenceDevice.AUTO
 *     decoder = Yolo26EndToEndDecoder() // Or AutoDetectionDecoder()
 *     classAgnosticNms = false          // Class-aware NMS preserves different overlapping classes
 *     numThreads = 4
 * }
 * ```
 */
class DetectorConfig {
    /** The TFLite model source. Required. */
    lateinit var model: ModelSource

    /** Labels source. Optional — if not set, labels will be "class_0", "class_1", etc. */
    var labels: LabelSource? = null

    /** Minimum confidence to keep a detection. Default: 0.25 (YOLO standard). */
    var confidenceThreshold: Float = 0.25f

    /** IoU threshold for Non-Maximum Suppression. Default: 0.45 */
    var iouThreshold: Float = 0.45f

    /** Maximum number of detections to return. Default: 50 */
    var maxResults: Int = 50

    /**
     * Whether Non-Maximum Suppression should suppress overlapping boxes across different classes.
     * Default: false (Class-Aware NMS: person and bicycle in same area both survive).
     */
    var classAgnosticNms: Boolean = false

    /**
     * Output decoder to parse model tensors.
     * Default: [AutoDetectionDecoder] which automatically identifies YOLO26 E2E, YOLOv8/v11/v26 Raw, and YOLOv5.
     */
    var decoder: DetectionDecoder = AutoDetectionDecoder()

    /** Hardware acceleration target. Default: AUTO (GPU with FP16 if available, otherwise XNNPACK CPU) */
    var device: InferenceDevice = InferenceDevice.AUTO

    /** Number of CPU threads for inference. Default: 4 */
    var numThreads: Int = 4

    /** Enable FP16 half-precision math for GPU acceleration (2-4x faster). Default: true */
    var useFp16: Boolean = true

    /**
     * Optional model input width override in pixels.
     * If null (default), auto-detected from model tensor shape.
     */
    var inputWidth: Int? = null

    /**
     * Optional model input height override in pixels.
     * If null (default), auto-detected from model tensor shape.
     */
    var inputHeight: Int? = null

    /**
     * Convenience model input size override for square models.
     * Sets both [inputWidth] and [inputHeight].
     */
    var inputSize: Int?
        get() = inputWidth ?: inputHeight
        set(value) {
            inputWidth = value
            inputHeight = value
        }

    /**
     * Enable exponential moving average (EMA) box smoothing.
     * Note: for decoupled stateless detection, smoothing can also be configured on [com.easyml.camera.EasyMLCameraView]
     * or via [DetectionSmoother].
     */
    var enableSmoothing: Boolean = true

    /**
     * Smoothing factor for EMA box smoothing [0.0 - 1.0].
     * Higher = more responsive to fast motion; Lower = smoother/steadier boxes. Default: 0.35
     */
    var smoothingFactor: Float = 0.35f

    internal fun validate() {
        check(::model.isInitialized) { "EasyML: 'model' must be set in DetectorConfig" }
        check(confidenceThreshold in 0f..1f) { "EasyML: confidenceThreshold must be in [0, 1]" }
        check(iouThreshold in 0f..1f) { "EasyML: iouThreshold must be in [0, 1]" }
        check(maxResults > 0) { "EasyML: maxResults must be > 0" }
        check(smoothingFactor in 0f..1f) { "EasyML: smoothingFactor must be in [0, 1]" }
        if (inputWidth != null) check(inputWidth!! > 0) { "EasyML: inputWidth must be > 0" }
        if (inputHeight != null) check(inputHeight!! > 0) { "EasyML: inputHeight must be > 0" }
    }
}
