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
    private val nms = NonMaxSuppression(config.iouThreshold, config.maxResults, config.classAgnosticNms)
    private val smoother: DetectionSmoother? = if (config.enableSmoothing) DetectionSmoother(config.smoothingFactor) else null

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
    private val outputByteBuffer: ByteBuffer
    private val outputBuffer: FloatArray
    private val outputShape: IntArray

    // Pre-allocated drawing structures for letterboxing
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
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
    fun detect(bitmap: Bitmap): List<Detection> {
        val results = ArrayList<Detection>(nms.maxResults)
        detect(bitmap, results)
        return results
    }

    /**
     * Zero-allocation detection overload that populates an existing [MutableList].
     *
     * Maximizes real-time performance in high-throughput video pipelines by reusing destination memory.
     */
    fun detect(bitmap: Bitmap, outResults: MutableList<Detection>) {
        outResults.clear()
        val t0 = System.nanoTime()

        // 1. Preprocessing: Letterbox aspect-ratio preservation
        val srcWidth = bitmap.width
        val srcHeight = bitmap.height
        val scale = min(inputWidth.toFloat() / srcWidth, inputHeight.toFloat() / srcHeight)
        val scaledWidth = srcWidth * scale
        val scaledHeight = srcHeight * scale
        val padX = (inputWidth - scaledWidth) / 2f
        val padY = (inputHeight - scaledHeight) / 2f

        letterboxCanvas.drawColor(0xFF808080.toInt()) // Standard neutral gray letterbox padding
        drawMatrix.reset()
        drawMatrix.setScale(scale, scale)
        drawMatrix.postTranslate(padX, padY)
        letterboxCanvas.drawBitmap(bitmap, drawMatrix, letterboxPaint)

        // Extract pixels
        letterboxBitmap.getPixels(pixelArray, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // Normalize pixels into inputByteBuffer
        inputByteBuffer.rewind()
        val numPixels = inputWidth * inputHeight

        if (isFloatType && inputFloatArray != null) {
            val inv255 = 1f / 255f
            if (isNCHW) {
                val plane2 = numPixels * 2
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    inputFloatArray[i] = ((pixel shr 16) and 0xFF) * inv255
                    inputFloatArray[numPixels + i] = ((pixel shr 8) and 0xFF) * inv255
                    inputFloatArray[plane2 + i] = (pixel and 0xFF) * inv255
                }
            } else {
                var offset = 0
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    inputFloatArray[offset++] = ((pixel shr 16) and 0xFF) * inv255
                    inputFloatArray[offset++] = ((pixel shr 8) and 0xFF) * inv255
                    inputFloatArray[offset++] = (pixel and 0xFF) * inv255
                }
            }
            inputByteBuffer.asFloatBuffer().put(inputFloatArray)
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

        // Decode candidates into reusable pool
        candidatePool.clear()
        decoder.decode(outputBuffer, outputShape, inputWidth, inputHeight, 0.25f, candidatePool)

        // Run NMS if the decoder is not already End-to-End NMS-free
        nmsPool.clear()
        if (decoder.isNmsFree) {
            val count = min(candidatePool.size, nms.maxResults)
            for (i in 0 until count) {
                nmsPool.add(candidatePool[i])
            }
        } else {
            nms.process(candidatePool, nmsPool)
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
    suspend fun detectAsync(bitmap: Bitmap): List<Detection> = withContext(Dispatchers.Default) {
        detect(bitmap)
    }

    /**
     * Non-blocking asynchronous detection populating a reusable [MutableList] on [Dispatchers.Default].
     */
    suspend fun detectAsync(bitmap: Bitmap, outResults: MutableList<Detection>) = withContext(Dispatchers.Default) {
        detect(bitmap, outResults)
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
}
