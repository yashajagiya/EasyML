package com.easyml.raw

import com.easyml.core.InferenceDevice
import com.easyml.core.ModelSource

/**
 * DSL configuration for [RawRunner].
 *
 * Usage:
 * ```kotlin
 * EasyML.raw(context) {
 *     model = ModelSource.Asset("custom_model.tflite")
 *     device = InferenceDevice.CPU
 *     numThreads = 4
 * }
 * ```
 */
class RawConfig {
    /** The TFLite model source. Required. */
    lateinit var model: ModelSource

    /** Hardware acceleration target. Default: AUTO */
    var device: InferenceDevice = InferenceDevice.AUTO

    /** Number of CPU threads. Default: 4 */
    var numThreads: Int = 4

    internal fun validate() {
        check(::model.isInitialized) { "EasyML: 'model' must be set in RawConfig" }
    }
}
