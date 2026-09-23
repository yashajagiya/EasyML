@file:Suppress("unused")

package com.easyml

import android.content.Context
import com.easyml.classification.ClassifierConfig
import com.easyml.classification.ImageClassifier
import com.easyml.detection.DetectorConfig
import com.easyml.detection.ObjectDetector
import com.easyml.raw.RawConfig
import com.easyml.raw.RawRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * # EasyML — Android-first Low-Overhead TFLite Engine
 *
 * Entry point for all EasyML functionality. Create detectors, classifiers,
 * or raw model runners with a simple Kotlin DSL.
 *
 * ## Object Detection (YOLO26, YOLO11, YOLOv8, YOLOv5)
 * ```kotlin
 * val detector = EasyML.objectDetector(context) {
 *     model = ModelSource.Asset("yolo26n.tflite")
 *     labels = LabelSource.Asset("labels.txt")
 *     confidenceThreshold = 0.35f
 *     device = InferenceDevice.AUTO
 * }
 * val results = detector.detect(bitmap)
 * ```
 *
 * ## Non-Blocking Asynchronous Model Loading (Coroutines)
 * ```kotlin
 * lifecycleScope.launch {
 *     val detector = EasyML.loadDetectorAsync(context) {
 *         model = ModelSource.Asset("yolo26n.tflite")
 *     }
 * }
 * ```
 */
object EasyML {

    /**
     * Create an [ObjectDetector] with DSL configuration synchronously.
     */
    fun objectDetector(
        context: Context,
        config: DetectorConfig.() -> Unit
    ): ObjectDetector {
        val cfg = DetectorConfig().apply(config)
        return ObjectDetector(context, cfg)
    }

    /**
     * Convenient shortcut alias for [objectDetector].
     */
    fun detector(
        context: Context,
        config: DetectorConfig.() -> Unit
    ): ObjectDetector = objectDetector(context, config)

    /**
     * Asynchronously load and initialize an [ObjectDetector] on [Dispatchers.IO]
     * to eliminate app launch ANRs and UI frame drops.
     */
    suspend fun loadDetectorAsync(
        context: Context,
        config: DetectorConfig.() -> Unit
    ): ObjectDetector = withContext(Dispatchers.IO) {
        objectDetector(context, config)
    }

    /**
     * Create an [ImageClassifier] with DSL configuration synchronously.
     */
    fun classifier(
        context: Context,
        config: ClassifierConfig.() -> Unit
    ): ImageClassifier {
        val cfg = ClassifierConfig().apply(config)
        return ImageClassifier(context, cfg)
    }

    /**
     * Asynchronously load and initialize an [ImageClassifier] on [Dispatchers.IO].
     */
    suspend fun loadClassifierAsync(
        context: Context,
        config: ClassifierConfig.() -> Unit
    ): ImageClassifier = withContext(Dispatchers.IO) {
        classifier(context, config)
    }

    /**
     * Create a [RawRunner] with DSL configuration.
     * Universal runner for ANY TFLite model with custom pre/post-processing.
     */
    fun raw(
        context: Context,
        config: RawConfig.() -> Unit
    ): RawRunner {
        val cfg = RawConfig().apply(config)
        return RawRunner(context, cfg)
    }
}
