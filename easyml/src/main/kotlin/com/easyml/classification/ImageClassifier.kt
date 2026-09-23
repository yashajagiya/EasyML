@file:Suppress("unused")

package com.easyml.classification

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
import java.nio.FloatBuffer
import kotlin.math.exp
import androidx.core.graphics.createBitmap

/**
 * High-level image classifier API optimized for mobile speed and low battery load.
 *
 * Performance features:
 * - Pre-allocated direct ByteBuffers (zero memory allocations per frame)
 * - Hardware acceleration (GPU FP16 / XNNPACK CPU)
 *
 * Usage:
 * ```kotlin
 * val classifier = EasyML.classifier(context) {
 *     model = ModelSource.Asset("mobilenet_v2.tflite")
 *     labels = LabelSource.Asset("imagenet_labels.txt")
 *     device = InferenceDevice.AUTO
 * }
 *
 * val results = classifier.classify(bitmap)
 * classifier.close()
 * ```
 */
@Stable
class ImageClassifier internal constructor(
    context: Context,
    private val config: ClassifierConfig
) : Closeable {

    private val engine: TFLiteEngine
    private val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int
    private val numClasses: Int
    private val isFloatType: Boolean
    private val isNCHW: Boolean
    private val channels: Int

    // Pre-allocated direct ByteBuffers reused every frame — zero allocations during inference!
    private val inputByteBuffer: ByteBuffer
    private val inputFloatBuffer: FloatBuffer?
    private val inputFloatArray: FloatArray?
    private val inputByteArray: ByteArray?
    private val outputByteBuffer: ByteBuffer
    private val outputFloatBuffer: FloatBuffer
    private val outputBuffer: FloatArray

    // Pre-allocated drawing structures for resizing
    private val resizedBitmap: Bitmap
    private val resizeCanvas: Canvas
    private val resizePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixelArray: IntArray
    private val resizeMatrix = Matrix()

    init {
        config.validate()

        engine = TFLiteEngine(
            context = context,
            modelSource = config.model,
            device = config.device,
            numThreads = config.numThreads
        )

        val inputShape = engine.inputShape // [1, height, width, 3] or [1, 3, height, width]
        val isNCHW = inputShape.size == 4 && inputShape[1] in 1..4 && inputShape.last() > 4
        this.isNCHW = isNCHW
        if (isNCHW) {
            channels = inputShape[1]
            inputHeight = inputShape[2]
            inputWidth = inputShape[3]
        } else if (inputShape.size == 4) {
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
            channels = inputShape[3]
        } else {
            inputHeight = inputShape.getOrElse(1) { 224 }
            inputWidth = inputShape.getOrElse(2) { 224 }
            channels = 3
        }

        numClasses = engine.outputShape.last()
        isFloatType = engine.inputDataType == DataType.FLOAT32

        outputBuffer = FloatArray(numClasses)
        outputByteBuffer = ByteBuffer.allocateDirect(numClasses * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        outputFloatBuffer = outputByteBuffer.asFloatBuffer()

        val bytesPerChannel = if (isFloatType) 4 else 1
        val totalInputElements = inputHeight * inputWidth * channels
        inputByteBuffer = ByteBuffer.allocateDirect(1 * totalInputElements * bytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }
        inputFloatBuffer = if (isFloatType) inputByteBuffer.asFloatBuffer() else null
        inputFloatArray = if (isFloatType) FloatArray(totalInputElements) else null
        inputByteArray = if (!isFloatType) ByteArray(totalInputElements) else null

        resizedBitmap = createBitmap(inputWidth, inputHeight)
        resizeCanvas = Canvas(resizedBitmap)
        pixelArray = IntArray(inputWidth * inputHeight)

        labels = config.labels?.resolve(context)
            ?: List(numClasses) { "class_$it" }
    }

    /**
     * Classify an image.
     *
     * Zero-allocation execution: reuses direct memory buffers and avoids GC pauses.
     *
     * @param bitmap Input image (any size — resized into pre-allocated memory)
     * @return Top-K classifications sorted by confidence (highest first)
     */
    fun classify(bitmap: Bitmap): List<Classification> {
        // 1. Resize into pre-allocated buffer
        resizeMatrix.reset()
        resizeMatrix.postScale(inputWidth.toFloat() / bitmap.width, inputHeight.toFloat() / bitmap.height)
        resizeCanvas.drawBitmap(bitmap, resizeMatrix, resizePaint)

        // 2. Extract pixels
        resizedBitmap.getPixels(pixelArray, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // 3. Fill input buffer
        inputByteBuffer.rewind()
        val numPixels = inputWidth * inputHeight

        if (isFloatType && inputFloatArray != null && inputFloatBuffer != null) {
            val limit = numPixels - 3
            var i = 0
            if (isNCHW) {
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

                    inputFloatArray[numPixels + i] = NORM_TABLE[(p0 ushr 8) and 0xFF]
                    inputFloatArray[numPixels + i + 1] = NORM_TABLE[(p1 ushr 8) and 0xFF]
                    inputFloatArray[numPixels + i + 2] = NORM_TABLE[(p2 ushr 8) and 0xFF]
                    inputFloatArray[numPixels + i + 3] = NORM_TABLE[(p3 ushr 8) and 0xFF]

                    inputFloatArray[plane2 + i] = NORM_TABLE[p0 and 0xFF]
                    inputFloatArray[plane2 + i + 1] = NORM_TABLE[p1 and 0xFF]
                    inputFloatArray[plane2 + i + 2] = NORM_TABLE[p2 and 0xFF]
                    inputFloatArray[plane2 + i + 3] = NORM_TABLE[p3 and 0xFF]

                    i += 4
                }
                while (i < numPixels) {
                    val p = pixelArray[i]
                    inputFloatArray[i] = NORM_TABLE[(p ushr 16) and 0xFF]
                    inputFloatArray[numPixels + i] = NORM_TABLE[(p ushr 8) and 0xFF]
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
            for (pixel in pixelArray) {
                inputByteArray[offset++] = ((pixel ushr 16) and 0xFF).toByte()
                inputByteArray[offset++] = ((pixel ushr 8) and 0xFF).toByte()
                inputByteArray[offset++] = (pixel and 0xFF).toByte()
            }
            inputByteBuffer.put(inputByteArray)
        }
        inputByteBuffer.rewind()

        // 4. Run hardware-accelerated inference
        outputByteBuffer.rewind()
        engine.run(inputByteBuffer, outputByteBuffer)

        // 5. Transfer outputs using pre-allocated FloatBuffer
        outputFloatBuffer.rewind()
        outputFloatBuffer.get(outputBuffer)

        // 6. Apply softmax
        val softmaxed = softmax(outputBuffer)

        // 7. Map to labels, filter, sort, and take top-K
        return softmaxed
            .mapIndexed { index, score ->
                Classification(
                    label = if (index < labels.size) labels[index] else "class_$index",
                    labelIndex = index,
                    confidence = score
                )
            }
            .filter { it.confidence >= config.confidenceThreshold }
            .sortedByDescending { it.confidence }
            .take(config.maxResults)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - maxLogit).toDouble()).toFloat() }
        val sumExp = exps.sum()
        return FloatArray(logits.size) { exps[it] / sumExp }
    }

    /**
     * Non-blocking asynchronous classification powered by Kotlin Coroutines on [Dispatchers.Default].
     */
    suspend fun classifyAsync(bitmap: Bitmap): List<Classification> = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        classify(bitmap)
    }

    override fun close() {
        engine.close()
        if (!resizedBitmap.isRecycled) {
            resizedBitmap.recycle()
        }
    }

    companion object {
        private val NORM_TABLE = FloatArray(256) { it / 255f }
    }
}
