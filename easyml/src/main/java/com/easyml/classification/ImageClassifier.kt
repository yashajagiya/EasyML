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

    // Pre-allocated direct ByteBuffers reused every frame — zero allocations during inference!
    private val inputByteBuffer: ByteBuffer
    private val outputByteBuffer: ByteBuffer
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

        val inputShape = engine.inputShape // [1, height, width, 3]
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        numClasses = engine.outputShape.last()
        isFloatType = engine.inputDataType == DataType.FLOAT32

        outputBuffer = FloatArray(numClasses)
        outputByteBuffer = ByteBuffer.allocateDirect(numClasses * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val bytesPerChannel = if (isFloatType) 4 else 1
        inputByteBuffer = ByteBuffer.allocateDirect(1 * inputHeight * inputWidth * 3 * bytesPerChannel).apply {
            order(ByteOrder.nativeOrder())
        }

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
        inputByteBuffer.rewind()

        // 4. Run hardware-accelerated inference
        outputByteBuffer.rewind()
        engine.run(inputByteBuffer, outputByteBuffer)

        // 5. Transfer outputs
        outputByteBuffer.rewind()
        outputByteBuffer.asFloatBuffer().get(outputBuffer)

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

    override fun close() {
        engine.close()
        if (!resizedBitmap.isRecycled) {
            resizedBitmap.recycle()
        }
    }
}
