@file:Suppress("unused")

package com.easyml.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import androidx.compose.runtime.Stable
import androidx.core.graphics.createBitmap
import com.easyml.core.TFLiteEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * High-performance, low-allocation Object Detection engine.
 *
 * Core architecture:
 * - **Deterministic & Stateless**: Clean separation between inference and optional temporal filters
 * - **Modular Decoders**: Built-in contracts for YOLO26 (Raw & E2E), YOLOv8, YOLO11, and YOLOv5
 * - **Class-Aware NMS**: Eliminates multi-class suppression bugs (person overlapping bicycle)
 * - **Quantization-Safe**: Automatic dequantization for INT8 and UINT8 models
 * - **High-Precision Metrics**: Microsecond profiling via [InferenceMetrics]
 * - **First-Class Coroutines**: Non-blocking [detectAsync] on Dispatchers.Default
 *
 * Usage:
 * ```kotlin
 * val detector = EasyML.objectDetector(context) {
 *     model = ModelSource.Asset("yolo26n.tflite")
 *     labels = LabelSource.Asset("labels.txt")
 *     device = InferenceDevice.AUTO
 * }
 *
 * val detections = detector.detect(bitmap)
 * detector.close()
 * ```
 */
@Stable
class ObjectDetector internal constructor(
    context: Context,
    config: DetectorConfig
) : Closeable {

    private val engine: TFLiteEngine
    private val decoder: DetectionDecoder = config.decoder
    private val confidenceThreshold: Float = config.confidenceThreshold
    private val nms = NonMaxSuppression(config.iouThreshold, config.maxResults, config.classAgnosticNms)
    private val smoother: DetectionSmoother? = if (config.enableSmoothing) DetectionSmoother(config.smoothingFactor) else null

    /** Whether internal temporal smoothing is active on this detector */
    val isSmoothingEnabled: Boolean get() = smoother != null

    /** Model input width in pixels */
    val inputWidth: Int

    /** Model input height in pixels */
    val inputHeight: Int

    private val labels: List<String>
    private val isFloatType: Boolean
    private val isNCHW: Boolean
    private val channels: Int

    // Pre-allocated ByteBuffers reused every frame to eliminate GC pauses
    private val inputByteBuffer: ByteBuffer
    private val inputFloatBuffer: FloatBuffer?
    private val outputByteBuffer: ByteBuffer
    private val outputBuffer: FloatArray
    private val outputShape: IntArray

    // Pre-allocated drawing structures for letterboxing
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val padPaint = Paint().apply { color = 0xFF808080.toInt() }
    private val pixelArray: IntArray
    private val inputFloatArray: FloatArray?
    private val inputByteArray: ByteArray?
    private val drawMatrix = Matrix()

    // Pre-allocated candidate pools reused every frame to eliminate heap churn
    private val candidatePool = ArrayList<DetectionCandidate>(500)
    private val nmsPool = ArrayList<DetectionCandidate>(config.maxResults)

    /** High-precision latency breakdown of the most recent frame */
    var lastMetrics: InferenceMetrics = InferenceMetrics()
        private set

    /** Total latency of the most recent execution in milliseconds (backward compatibility) */
    val lastInferenceTimeMs: Long
        get() = lastMetrics.totalMs.toLong()

    init {
        config.validate()

        engine = TFLiteEngine(
            context = context,
            modelSource = config.model,
            device = config.device,
            numThreads = config.numThreads,
            useFp16 = config.useFp16
        )

        val modelInputShape = engine.inputShape
        val isNCHW = modelInputShape.size == 4 && modelInputShape[1] in 1..4 && modelInputShape.last() > 4
        val (modelHeight, modelWidth, channels) = if (isNCHW) {
            Triple(modelInputShape[2], modelInputShape[3], modelInputShape[1])
        } else if (modelInputShape.size == 4) {
            Triple(modelInputShape[1], modelInputShape[2], modelInputShape[3])
        } else {
            Triple(modelInputShape.getOrElse(1) { 640 }, modelInputShape.getOrElse(2) { 640 }, 3)
        }

        this.inputWidth = config.inputWidth ?: modelWidth
        this.inputHeight = config.inputHeight ?: modelHeight
        this.isNCHW = isNCHW
        this.channels = channels
        this.isFloatType = engine.inputDataType == DataType.FLOAT32

        // Output shape & buffer pre-allocation
        outputShape = engine.outputShape
        val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
        outputBuffer = FloatArray(outputSize)

        val outputBytesPerChannel = when (engine.outputDataType) {
            DataType.FLOAT32 -> 4
            DataType.INT8, DataType.UINT8 -> 1
            else -> 4
        }
        outputByteBuffer = ByteBuffer.allocateDirect(outputSize * outputBytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }

        // Input buffer pre-allocation
        val inputBytesPerChannel = if (isFloatType) 4 else 1
        val totalInputElements = inputWidth * inputHeight * channels
        inputByteBuffer = ByteBuffer.allocateDirect(1 * totalInputElements * inputBytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }
        inputFloatBuffer = if (isFloatType) inputByteBuffer.asFloatBuffer() else null

        inputFloatArray = if (isFloatType) FloatArray(totalInputElements) else null
        inputByteArray = if (!isFloatType) ByteArray(totalInputElements) else null

        // Letterbox drawing buffer
        letterboxBitmap = createBitmap(inputWidth, inputHeight)
        letterboxCanvas = Canvas(letterboxBitmap)
        pixelArray = IntArray(inputWidth * inputHeight)

        // Load labels
        labels = config.labels?.resolve(context)
            ?: List(
                when (outputShape.size) {
                    3 if outputShape[2] == 6 -> 80 // YOLO26 E2E default
                    3 if outputShape[1] < outputShape[2] -> outputShape[1] - 4
                    3 -> outputShape[2] - 4
                    else -> outputShape.last() - 4
                }.coerceAtLeast(1)
            ) { "class_$it" }
    }

    /**
     * Run object detection synchronously on a Bitmap.
     *
     * @param bitmap Input image (resized and letterboxed into pre-allocated memory)
     * @return List of detected objects, sorted by confidence (highest first)
     */
    /**
     * Run object detection synchronously on a Bitmap.
     *
     * @param bitmap Input image (resized and letterboxed into pre-allocated memory)
     * @param rotationDegrees Optional camera sensor rotation (e.g. 90, 270) handled in a single hardware Skia pass
     * @return List of detected objects, sorted by confidence (highest first)
     */
    fun detect(bitmap: Bitmap, rotationDegrees: Int = 0): List<Detection> {
        val results = ArrayList<Detection>(nms.maxResults)
        detect(bitmap, rotationDegrees, results)
        return results
    }

    /**
     * Zero-allocation detection overload that populates an existing [MutableList].
     *
     * Maximizes real-time performance in high-throughput video pipelines by reusing destination memory.
     */
    fun detect(bitmap: Bitmap, outResults: MutableList<Detection>) {
        detect(bitmap, 0, outResults)
    }

    /**
     * Zero-allocation detection overload with hardware matrix rotation.
     */
    fun detect(bitmap: Bitmap, rotationDegrees: Int, outResults: MutableList<Detection>) {
        outResults.clear()
        val t0 = System.nanoTime()

        // 1. Preprocessing: Letterbox aspect-ratio preservation with optional Skia hardware rotation
        val isRotated = rotationDegrees == 90 || rotationDegrees == 270
        val srcWidth = if (isRotated) bitmap.height else bitmap.width
        val srcHeight = if (isRotated) bitmap.width else bitmap.height
        val scale = min(inputWidth.toFloat() / srcWidth, inputHeight.toFloat() / srcHeight)
        val scaledWidth = srcWidth * scale
        val scaledHeight = srcHeight * scale
        val padX = (inputWidth - scaledWidth) * 0.5f
        val padY = (inputHeight - scaledHeight) * 0.5f

        // Smart letterbox fill: only draw border margins if padding is present
        if (padX > 0f) {
            letterboxCanvas.drawRect(0f, 0f, padX, inputHeight.toFloat(), padPaint)
            letterboxCanvas.drawRect(inputWidth - padX, 0f, inputWidth.toFloat(), inputHeight.toFloat(), padPaint)
        }
        if (padY > 0f) {
            letterboxCanvas.drawRect(0f, 0f, inputWidth.toFloat(), padY, padPaint)
            letterboxCanvas.drawRect(0f, inputHeight - padY, inputWidth.toFloat(), inputHeight.toFloat(), padPaint)
        }

        drawMatrix.reset()
        if (rotationDegrees != 0) {
            drawMatrix.postTranslate(-bitmap.width * 0.5f, -bitmap.height * 0.5f)
            drawMatrix.postRotate(rotationDegrees.toFloat())
            drawMatrix.postScale(scale, scale)
            drawMatrix.postTranslate(inputWidth * 0.5f, inputHeight * 0.5f)
        } else {
            drawMatrix.setScale(scale, scale)
            drawMatrix.postTranslate(padX, padY)
        }
        letterboxCanvas.drawBitmap(bitmap, drawMatrix, letterboxPaint)

        // Extract pixels
        letterboxBitmap.getPixels(pixelArray, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // Normalize pixels into inputByteBuffer
        inputByteBuffer.rewind()
        val numPixels = inputWidth * inputHeight

        if (isFloatType && inputFloatArray != null && inputFloatBuffer != null) {
            val limit = numPixels - 3
            var i = 0
            if (isNCHW) {
                val plane1 = numPixels
                val plane2 = numPixels * 2
                while (i < limit) {
                    val p0 = pixelArray[i]
                    val p1 = pixelArray[i + 1]
                    val p2 = pixelArray[i + 2]
                    val p3 = pixelArray[i + 3]

                    inputFloatArray[i] = NORM_TABLE[(p0 ushr 16) and 0xFF]
                    inputFloatArray[i + 1] = NORM_TABLE[(p1 ushr 16) and 0xFF]
                    inputFloatArray[i + 2] = NORM_TABLE[(p2 ushr 16) and 0xFF]
                    inputFloatArray[i + 3] = NORM_TABLE[(p3 ushr 16) and 0xFF]

                    inputFloatArray[plane1 + i] = NORM_TABLE[(p0 ushr 8) and 0xFF]
                    inputFloatArray[plane1 + i + 1] = NORM_TABLE[(p1 ushr 8) and 0xFF]
                    inputFloatArray[plane1 + i + 2] = NORM_TABLE[(p2 ushr 8) and 0xFF]
                    inputFloatArray[plane1 + i + 3] = NORM_TABLE[(p3 ushr 8) and 0xFF]

                    inputFloatArray[plane2 + i] = NORM_TABLE[p0 and 0xFF]
                    inputFloatArray[plane2 + i + 1] = NORM_TABLE[p1 and 0xFF]
                    inputFloatArray[plane2 + i + 2] = NORM_TABLE[p2 and 0xFF]
                    inputFloatArray[plane2 + i + 3] = NORM_TABLE[p3 and 0xFF]

                    i += 4
                }
                while (i < numPixels) {
                    val p = pixelArray[i]
                    inputFloatArray[i] = NORM_TABLE[(p ushr 16) and 0xFF]
                    inputFloatArray[plane1 + i] = NORM_TABLE[(p ushr 8) and 0xFF]
                    inputFloatArray[plane2 + i] = NORM_TABLE[p and 0xFF]
                    i++
                }
            } else {
                var offset = 0
                while (i < limit) {
                    val p0 = pixelArray[i]
                    val p1 = pixelArray[i + 1]
                    val p2 = pixelArray[i + 2]
                    val p3 = pixelArray[i + 3]

                    inputFloatArray[offset] = NORM_TABLE[(p0 ushr 16) and 0xFF]
                    inputFloatArray[offset + 1] = NORM_TABLE[(p0 ushr 8) and 0xFF]
                    inputFloatArray[offset + 2] = NORM_TABLE[p0 and 0xFF]

                    inputFloatArray[offset + 3] = NORM_TABLE[(p1 ushr 16) and 0xFF]
                    inputFloatArray[offset + 4] = NORM_TABLE[(p1 ushr 8) and 0xFF]
                    inputFloatArray[offset + 5] = NORM_TABLE[p1 and 0xFF]

                    inputFloatArray[offset + 6] = NORM_TABLE[(p2 ushr 16) and 0xFF]
                    inputFloatArray[offset + 7] = NORM_TABLE[(p2 ushr 8) and 0xFF]
                    inputFloatArray[offset + 8] = NORM_TABLE[p2 and 0xFF]

                    inputFloatArray[offset + 9] = NORM_TABLE[(p3 ushr 16) and 0xFF]
                    inputFloatArray[offset + 10] = NORM_TABLE[(p3 ushr 8) and 0xFF]
                    inputFloatArray[offset + 11] = NORM_TABLE[p3 and 0xFF]

                    offset += 12
                    i += 4
                }
                while (i < numPixels) {
                    val p = pixelArray[i]
                    inputFloatArray[offset++] = NORM_TABLE[(p ushr 16) and 0xFF]
                    inputFloatArray[offset++] = NORM_TABLE[(p ushr 8) and 0xFF]
                    inputFloatArray[offset++] = NORM_TABLE[p and 0xFF]
                    i++
                }
            }
            inputFloatBuffer.rewind()
            inputFloatBuffer.put(inputFloatArray)
        } else if (inputByteArray != null) {
            var offset = 0
            for (i in 0 until numPixels) {
                val pixel = pixelArray[i]
                inputByteArray[offset++] = ((pixel shr 16) and 0xFF).toByte()
                inputByteArray[offset++] = ((pixel shr 8) and 0xFF).toByte()
                inputByteArray[offset++] = (pixel and 0xFF).toByte()
            }
            inputByteBuffer.put(inputByteArray)
        }
        inputByteBuffer.rewind()

        val t1 = System.nanoTime()

        // 2. Hardware Inference
        outputByteBuffer.rewind()
        engine.run(inputByteBuffer, outputByteBuffer)

        val t2 = System.nanoTime()

        // 3. Postprocessing: Read and dequantize tensor
        engine.readOutput(outputByteBuffer, outputBuffer)

        // Decode candidates into reusable pool (preserves pre-allocated objects without garbage collection)
        val candidateCount = decoder.decode(outputBuffer, outputShape, inputWidth, inputHeight, confidenceThreshold, candidatePool)

        // Run NMS if the decoder is not already End-to-End NMS-free
        nmsPool.clear()
        if (decoder.isNmsFree) {
            val count = minOf(candidateCount, nms.maxResults)
            for (i in 0 until count) {
                nmsPool.add(candidatePool[i])
            }
        } else {
            nms.process(candidatePool, candidateCount, nmsPool)
        }

        // Unmap coordinates from model space to original image space
        val invScale = 1f / scale
        for (i in nmsPool.indices) {
            val cand = nmsPool[i]
            val left = max(0f, (cand.left - padX) * invScale)
            val top = max(0f, (cand.top - padY) * invScale)
            val right = min(srcWidth.toFloat(), (cand.right - padX) * invScale)
            val bottom = min(srcHeight.toFloat(), (cand.bottom - padY) * invScale)

            val labelText = if (cand.labelIndex in labels.indices) labels[cand.labelIndex] else "class_${cand.labelIndex}"
            outResults.add(
                Detection(
                    boundingBox = RectF(left, top, right, bottom),
                    label = labelText,
                    labelIndex = cand.labelIndex,
                    confidence = cand.confidence
                )
            )
        }

        // Apply temporal smoothing if configured
        val finalDetections = smoother?.update(outResults) ?: outResults
        if (smoother != null) {
            outResults.clear()
            outResults.addAll(finalDetections)
        }

        val t3 = System.nanoTime()

        lastMetrics = InferenceMetrics(
            preprocessMs = (t1 - t0) / 1_000_000.0,
            inferenceMs = (t2 - t1) / 1_000_000.0,
            postprocessMs = (t3 - t2) / 1_000_000.0,
            totalMs = (t3 - t0) / 1_000_000.0
        )
    }

    /**
     * Non-blocking asynchronous detection powered by Kotlin Coroutines on [Dispatchers.Default].
     */
    suspend fun detectAsync(bitmap: Bitmap, rotationDegrees: Int = 0): List<Detection> = withContext(Dispatchers.Default) {
        val results = ArrayList<Detection>(nms.maxResults)
        detect(bitmap, rotationDegrees, results)
        results
    }

    /**
     * Non-blocking asynchronous detection populating a reusable [MutableList] on [Dispatchers.Default].
     */
    suspend fun detectAsync(
        bitmap: Bitmap,
        rotationDegrees: Int = 0,
        outResults: MutableList<Detection>
    ) = withContext(Dispatchers.Default) {
        detect(bitmap, rotationDegrees, outResults)
    }

    /**
     * Non-blocking asynchronous detection populating a reusable [MutableList] on [Dispatchers.Default].
     */
    suspend fun detectAsync(bitmap: Bitmap, outResults: MutableList<Detection>) = withContext(Dispatchers.Default) {
        detect(bitmap, 0, outResults)
    }

    /** Get the primary model input dimension (backward compatibility). */
    fun getInputSize(): Int = max(inputWidth, inputHeight)

    /** Get the loaded class labels. */
    fun getLabels(): List<String> = labels.toList()

    override fun close() {
        engine.close()
        if (!letterboxBitmap.isRecycled) {
            letterboxBitmap.recycle()
        }
    }

    companion object {
        private val NORM_TABLE = FloatArray(256) { it / 255f }
    }
}
