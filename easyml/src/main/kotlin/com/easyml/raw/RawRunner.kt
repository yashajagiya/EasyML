@file:Suppress("unused")

package com.easyml.raw

import android.content.Context
import androidx.compose.runtime.Stable
import com.easyml.core.TFLiteEngine
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Universal raw model runner for ANY TFLite model.
 * No assumptions about input/output format — you handle pre/post-processing.
 *
 * Usage:
 * ```kotlin
 * val runner = EasyML.raw(context) {
 *     model = ModelSource.Asset("custom_model.tflite")
 * }
 *
 * // Single input/output
 * val input: ByteBuffer = ... // Your preprocessed input
 * val output: ByteBuffer = ByteBuffer.allocateDirect(outputSize)
 * runner.run(input, output)
 *
 * // Multiple outputs
 * val outputs = mutableMapOf<Int, Any>(
 *     0 to FloatArray(100),
 *     1 to FloatArray(25)
 * )
 * runner.runMultiple(arrayOf(input), outputs)
 *
 * runner.close()
 * ```
 */
@Stable
class RawRunner internal constructor(
    context: Context,
    config: RawConfig
) : Closeable {

    private val engine: TFLiteEngine

    /** Input tensor shape */
    val inputShape: IntArray
        get() = engine.inputShape

    /** Output tensor shape (first output) */
    val outputShape: IntArray
        get() = engine.outputShape

    /** Number of output tensors */
    val outputCount: Int
        get() = engine.outputTensorCount

    init {
        config.validate()
        engine = TFLiteEngine(
            context = context,
            modelSource = config.model,
            device = config.device,
            numThreads = config.numThreads
        )
    }

    /**
     * Run inference with single input and output ByteBuffers.
     */
    fun run(input: ByteBuffer, output: ByteBuffer) {
        engine.run(input, output)
    }

    /**
     * Run inference with single input/output (any type TFLite supports).
     */
    fun run(input: Any, output: Any) {
        engine.run(input, output)
    }

    /**
     * Run inference with multiple inputs and outputs.
     *
     * @param inputs Array of input data
     * @param outputs Map of output index to output buffer (e.g., FloatArray, ByteBuffer)
     */
    fun runMultiple(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
        engine.runMultiOutput(inputs, outputs)
    }

    /**
     * Non-blocking asynchronous inference with ByteBuffers on [Dispatchers.Default].
     */
    suspend fun runAsync(input: ByteBuffer, output: ByteBuffer) = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        run(input, output)
    }

    /**
     * Non-blocking asynchronous inference with multiple inputs and outputs on [Dispatchers.Default].
     */
    suspend fun runMultipleAsync(inputs: Array<Any>, outputs: MutableMap<Int, Any>) = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        runMultiple(inputs, outputs)
    }

    /**
     * Get the shape of a specific output tensor.
     */
    fun getOutputShape(index: Int): IntArray = engine.getOutputShape(index)

    override fun close() {
        engine.close()
    }
}
