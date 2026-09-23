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

    // Pre-allocated letterboxing canvas and bitmap (allocated once, never recreated)
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val letterboxPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixelArray: IntArray
    private val drawMatrix = Matrix()

    init {
        config.validate()

        // Initialize the TFLite engine
        engine = TFLiteEngine(
            context = context,
            modelSource = config.model,
            device = config.device,
            numThreads = config.numThreads
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
        inputByteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * channels * bytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }

        // Pre-allocate letterbox drawing structures
        letterboxBitmap = createBitmap(inputSize, inputSize)
        letterboxCanvas = Canvas(letterboxBitmap)
        pixelArray = IntArray(inputSize * inputSize)

        // Initialize post-processor
        postProcessor = YoloPostProcessor(
            confidenceThreshold = config.confidenceThreshold,
            iouThreshold = config.iouThreshold,
            maxResults = config.maxResults,
            labels = labels
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

        // 3. Populate pre-allocated direct input buffer (fast normalization)
        inputByteBuffer.rewind()
        if (isNCHW) {
            // Planar format: RRR...GGG...BBB...
            if (isFloatType) {
                val inv255 = 1f / 255f
                for (pixel in pixelArray) {
                    inputByteBuffer.putFloat(((pixel shr 16) and 0xFF) * inv255)
                }
                for (pixel in pixelArray) {
                    inputByteBuffer.putFloat(((pixel shr 8) and 0xFF) * inv255)
                }
                for (pixel in pixelArray) {
                    inputByteBuffer.putFloat((pixel and 0xFF) * inv255)
                }
            } else {
                for (pixel in pixelArray) {
                    inputByteBuffer.put(((pixel shr 16) and 0xFF).toByte())
                }
                for (pixel in pixelArray) {
                    inputByteBuffer.put(((pixel shr 8) and 0xFF).toByte())
                }
                for (pixel in pixelArray) {
                    inputByteBuffer.put((pixel and 0xFF).toByte())
                }
            }
        } else {
            // Interleaved format: RGBRGBRGB...
            if (isFloatType) {
                val inv255 = 1f / 255f
                for (pixel in pixelArray) {
                    inputByteBuffer.putFloat(((pixel shr 16) and 0xFF) * inv255)
                    inputByteBuffer.putFloat(((pixel shr 8) and 0xFF) * inv255)
                    inputByteBuffer.putFloat((pixel and 0xFF) * inv255)
                }
            } else {
                for (pixel in pixelArray) {
                    inputByteBuffer.put(((pixel shr 16) and 0xFF).toByte())
                    inputByteBuffer.put(((pixel shr 8) and 0xFF).toByte())
                    inputByteBuffer.put((pixel and 0xFF).toByte())
                }
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
        return postProcessor.process(
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
