@file:Suppress("unused")

package com.easyml

import android.content.Context
import com.easyml.classification.ClassifierConfig
import com.easyml.classification.ImageClassifier
import com.easyml.detection.DetectorConfig
import com.easyml.detection.ObjectDetector
import com.easyml.raw.RawConfig
import com.easyml.raw.RawRunner

/**
 * # EasyML — Simple TFLite for Android
 *
 * Entry point for all EasyML functionality. Create detectors, classifiers,
 * or raw model runners with a simple Kotlin DSL.
 *
 * ## Object Detection (YOLO, SSD, etc.)
 * ```kotlin
 * val detector = EasyML.objectDetector(context) {
 *     model = ModelSource.Asset("yolon.tflite")
 *     labels = LabelSource.Asset("labels.txt")
 *     confidenceThreshold = 0.5f
 *     device = InferenceDevice.GPU
 * }
 * val results = detector.detect(bitmap)
 * ```
 *
 * ## Image Classification (MobileNet, EfficientNet, etc.)
 * ```kotlin
 * val classifier = EasyML.classifier(context) {
 *     model = ModelSource.Asset("mobilenet.tflite")
 *     labels = LabelSource.Asset("labels.txt")
 * }
 * val categories = classifier.classify(bitmap)
 * ```
 *
 * ## Raw Model (any .tflite)
 * ```kotlin
 * val runner = EasyML.raw(context) {
 *     model = ModelSource.Asset("custom.tflite")
 * }
 * runner.run(inputBuffer, outputBuffer)
 * ```
 *
 * ## Live Camera Detection (Compose)
 * ```kotlin
 * EasyMLCameraView(
 *     detector = detector,
 *     showOverlay = true,
 *     showFps = true
 * )
 * ```
 */
object EasyML {

    /**
     * Create an [ObjectDetector] with DSL configuration.
     *
     * Supports YOLO (v5, v8, v11, v26), SSD, and other detection models.
     * Automatically detects the output tensor format and applies appropriate post-processing.
     */
    fun objectDetector(
        context: Context,
        config: DetectorConfig.() -> Unit
    ): ObjectDetector {
        val cfg = DetectorConfig().apply(config)
        return ObjectDetector(context, cfg)
    }

    /**
     * Create an [ImageClassifier] with DSL configuration.
     *
     * Supports MobileNet, EfficientNet, ResNet, and any classification model
     * that outputs a 1D probability vector.
     */
    fun classifier(
        context: Context,
        config: ClassifierConfig.() -> Unit
    ): ImageClassifier {
        val cfg = ClassifierConfig().apply(config)
        return ImageClassifier(context, cfg)
    }

    /**
     * Create a [RawRunner] with DSL configuration.
     *
     * Universal runner for ANY TFLite model. No assumptions about input/output format.
     * You handle your own pre-processing and post-processing.
     */
    fun raw(
        context: Context,
        config: RawConfig.() -> Unit
    ): RawRunner {
        val cfg = RawConfig().apply(config)
        return RawRunner(context, cfg)
    }
}
