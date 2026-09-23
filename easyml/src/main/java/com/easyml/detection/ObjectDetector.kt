@file:Suppress("unused")

package com.easyml.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.compose.runtime.Stable
import com.easyml.core.TFLiteEngine
import org.tensorflow.lite.DataType
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.createBitmap

/**
 * High-level object detector API optimized for mobile performance.
 *
 * Performance features:
 * - Zero-allocation inference loop (pre-allocated direct ByteBuffers & working bitmaps)
 * - Cache-optimized sequential post-processing
 * - Automatic hardware acceleration (GPU FP16 / XNNPACK CPU)
 *
 * Usage:
 * ```kotlin
 * val detector = EasyML.objectDetector(context) {
 *     model = ModelSource.Asset("yolon.tflite")
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
    private val postProcessor: YoloPostProcessor
    private val inputSize: Int
    private val labels: List<String>
    private val isFloatType: Boolean
    private val isNCHW: Boolean
    private val channels: Int

    // Pre-allocated ByteBuffers reused every frame — zero allocations during inference!
    private val inputByteBuffer: ByteBuffer
    private val outputByteBuffer: ByteBuffer
    private val outputBuffer: FloatArray
    private val outputShape: IntArray

    // Pre-allocated drawing structures
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixelArray: IntArray
    private val inputFloatArray: FloatArray?
    private val inputByteArray: ByteArray?
    private val drawMatrix = Matrix()

    /** Latency of the most recent inference execution in milliseconds */
    var lastInferenceTimeMs: Long = 0L
        private set

    init {
        config.validate()

        // Initialize the TFLite engine with FP16 support
        engine = TFLiteEngine(
            context = context,
            modelSource = config.model,
            device = config.device,
            numThreads = config.numThreads,
            useFp16 = config.useFp16
        )

        // Auto-detect input size and layout from model
        val modelInputShape = engine.inputShape // e.g., [1, 3, 640, 640] (NCHW) or [1, 640, 640, 3] (NHWC)
        val isNCHW = modelInputShape.size == 4 && modelInputShape[1] in 1..4 && modelInputShape.last() > 4
        val (modelHeight, modelWidth, channels) = if (isNCHW) {
            Triple(modelInputShape[2], modelInputShape[3], modelInputShape[1])
        } else if (modelInputShape.size == 4) {
            Triple(modelInputShape[1], modelInputShape[2], modelInputShape[3])
        } else {
            Triple(modelInputShape.getOrElse(1) { 640 }, modelInputShape.getOrElse(2) { 640 }, 3)
        }

        inputSize = config.inputSize ?: maxOf(modelWidth, modelHeight)
        this.isNCHW = isNCHW
        this.channels = channels
        isFloatType = engine.inputDataType == DataType.FLOAT32

        // Load labels
        labels = config.labels?.resolve(context)
            ?: List(engine.outputShape.let { shape ->
                when (shape.size) {
                    3 if shape[1] < shape[2] -> shape[1] - 4  // v8 format
                    3 if true -> shape[2] - 4 // v5 format
                    else -> shape.last() - 4
                }
            }) { "class_$it" }

        // Pre-allocate output buffers
        outputShape = engine.outputShape
        val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
        outputBuffer = FloatArray(outputSize)
        outputByteBuffer = ByteBuffer.allocateDirect(outputSize * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        // Pre-allocate input buffer: 1 * H * W * C * bytesPerChannel
        val bytesPerChannel = if (isFloatType) 4 else 1
        val totalElements = inputSize * inputSize * channels
        inputByteBuffer = ByteBuffer.allocateDirect(1 * totalElements * bytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }

        // Pre-allocate bulk conversion arrays
        inputFloatArray = if (isFloatType) FloatArray(totalElements) else null
        inputByteArray = if (!isFloatType) ByteArray(totalElements) else null

        // Pre-allocate letterbox drawing structures
        letterboxBitmap = createBitmap(inputSize, inputSize)
        letterboxCanvas = Canvas(letterboxBitmap)
        pixelArray = IntArray(inputSize * inputSize)

        // Initialize post-processor with smoothing support
        postProcessor = YoloPostProcessor(
            confidenceThreshold = config.confidenceThreshold,
            iouThreshold = config.iouThreshold,
            maxResults = config.maxResults,
            labels = labels,
            enableSmoothing = config.enableSmoothing,
            smoothingFactor = config.smoothingFactor
        )
    }

    /**
     * Run object detection on a Bitmap.
     *
     * Zero-allocation execution: reuses direct memory buffers and avoids GC pauses.
     *
     * @param bitmap Input image (any size — resized and letterboxed into pre-allocated memory)
     * @return List of detected objects, sorted by confidence (highest first)
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val startTime = System.currentTimeMillis()
        val srcW = bitmap.width
        val srcH = bitmap.height
        val scale = minOf(inputSize.toFloat() / srcW, inputSize.toFloat() / srcH)
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        val padX = (inputSize - newW) / 2
        val padY = (inputSize - newH) / 2

        // 1. Draw scaled bitmap with letterbox padding into pre-allocated buffer
        letterboxCanvas.drawColor(0xFF808080.toInt()) // Standard YOLO gray background
        drawMatrix.reset()
        drawMatrix.postScale(scale, scale)
        drawMatrix.postTranslate(padX.toFloat(), padY.toFloat())
        letterboxCanvas.drawBitmap(bitmap, drawMatrix, letterboxPaint)

        // 2. Extract pixels into pre-allocated IntArray
        letterboxBitmap.getPixels(pixelArray, 0, inputSize, 0, 0, inputSize, inputSize)

        // 3. Fast bulk normalization directly into direct memory
        val numPixels = inputSize * inputSize
        if (isNCHW) {
            if (isFloatType) {
                val floatArray = inputFloatArray!!
                val inv255 = 1f / 255f
                val gOffset = numPixels
                val bOffset = numPixels * 2
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    floatArray[i] = ((pixel shr 16) and 0xFF) * inv255
                    floatArray[gOffset + i] = ((pixel shr 8) and 0xFF) * inv255
                    floatArray[bOffset + i] = (pixel and 0xFF) * inv255
                }
                inputByteBuffer.rewind()
                inputByteBuffer.asFloatBuffer().put(floatArray)
            } else {
                val byteArray = inputByteArray!!
                val gOffset = numPixels
                val bOffset = numPixels * 2
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    byteArray[i] = ((pixel shr 16) and 0xFF).toByte()
                    byteArray[gOffset + i] = ((pixel shr 8) and 0xFF).toByte()
                    byteArray[bOffset + i] = (pixel and 0xFF).toByte()
                }
                inputByteBuffer.rewind()
                inputByteBuffer.put(byteArray)
            }
        } else {
            // NHWC: Interleaved
            if (isFloatType) {
                val floatArray = inputFloatArray!!
                val inv255 = 1f / 255f
                var idx = 0
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    floatArray[idx++] = ((pixel shr 16) and 0xFF) * inv255
                    floatArray[idx++] = ((pixel shr 8) and 0xFF) * inv255
                    floatArray[idx++] = (pixel and 0xFF) * inv255
                }
                inputByteBuffer.rewind()
                inputByteBuffer.asFloatBuffer().put(floatArray)
            } else {
                val byteArray = inputByteArray!!
                var idx = 0
                for (i in 0 until numPixels) {
                    val pixel = pixelArray[i]
                    byteArray[idx++] = ((pixel shr 16) and 0xFF).toByte()
                    byteArray[idx++] = ((pixel shr 8) and 0xFF).toByte()
                    byteArray[idx++] = (pixel and 0xFF).toByte()
                }
                inputByteBuffer.rewind()
                inputByteBuffer.put(byteArray)
            }
        }
        inputByteBuffer.rewind()

        // 4. Run hardware-accelerated inference
        outputByteBuffer.rewind()
        engine.run(inputByteBuffer, outputByteBuffer)

        // 5. Transfer to float buffer for post-processing
        outputByteBuffer.rewind()
        outputByteBuffer.asFloatBuffer().get(outputBuffer)

        // 6. Post-process with cache-friendly algorithms
        val results = postProcessor.process(
            output = outputBuffer,
            outputShape = outputShape,
            originalWidth = srcW,
            originalHeight = srcH,
            letterboxScale = scale,
            letterboxPadX = padX,
            letterboxPadY = padY,
            modelWidth = inputSize,
            modelHeight = inputSize
        )

        lastInferenceTimeMs = System.currentTimeMillis() - startTime
        return results
    }

    /**
     * Get the model's expected input size.
     */
    fun getInputSize(): Int = inputSize

    /**
     * Get the loaded labels.
     */
    fun getLabels(): List<String> = labels.toList()

    override fun close() {
        engine.close()
        if (!letterboxBitmap.isRecycled) {
            letterboxBitmap.recycle()
        }
    }
}
