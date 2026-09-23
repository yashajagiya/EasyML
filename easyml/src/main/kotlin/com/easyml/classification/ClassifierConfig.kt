package com.easyml.classification

import com.easyml.core.InferenceDevice
import com.easyml.core.LabelSource
import com.easyml.core.ModelSource

/**
 * DSL configuration for [ImageClassifier].
 *
 * Usage:
 * ```kotlin
 * EasyML.classifier(context) {
 *     model = ModelSource.Asset("mobilenet.tflite")
 *     labels = LabelSource.Asset("imagenet_labels.txt")
 *     maxResults = 5
 *     confidenceThreshold = 0.1f
 * }
 * ```
 */
class ClassifierConfig {
    /** The TFLite model source. Required. */
    lateinit var model: ModelSource

    /** Labels source. Optional. */
    var labels: LabelSource? = null

    /** Maximum number of top classifications to return. Default: 5 */
    var maxResults: Int = 5

    /** Minimum confidence to include in results. Default: 0.01 */
    var confidenceThreshold: Float = 0.01f

    /** Hardware acceleration target. Default: AUTO */
    var device: InferenceDevice = InferenceDevice.AUTO

    /** Number of CPU threads. Default: 4 */
    var numThreads: Int = 4

    internal fun validate() {
        check(::model.isInitialized) { "EasyML: 'model' must be set in ClassifierConfig" }
    }
}
