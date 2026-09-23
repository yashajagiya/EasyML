package com.easyml.core

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer

/**
 * Core TFLite interpreter wrapper. Manages the interpreter lifecycle,
 * provides thread-safe inference, and handles resource cleanup.
 *
 * This is an internal component — consumers use [com.easyml.detection.ObjectDetector],
 * [com.easyml.classification.ImageClassifier], or [com.easyml.raw.RawRunner] instead.
 */
internal class TFLiteEngine(
    context: Context,
    modelSource: ModelSource,
    device: InferenceDevice = InferenceDevice.CPU,
    numThreads: Int = 4,
    useFp16: Boolean = true
) : Closeable {

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    private val lock = Any()

    /** Input tensor shape (e.g., [1, 640, 640, 3]) */
    val inputShape: IntArray
        get() = interpreter.getInputTensor(0).shape()

    /** Output tensor shape (e.g., [1, 84, 8400]) */
    val outputShape: IntArray
        get() = interpreter.getOutputTensor(0).shape()

    /** Number of output tensors */
    val outputTensorCount: Int
        get() = interpreter.outputTensorCount

    /** Get shape of a specific output tensor */
    fun getOutputShape(index: Int): IntArray =
        interpreter.getOutputTensor(index).shape()

    /** Get data type of input tensor */
    val inputDataType: org.tensorflow.lite.DataType
        get() = interpreter.getInputTensor(0).dataType()

    init {
        val modelBuffer = modelSource.resolve(context)
        var delegate: GpuDelegate? = null
        var interp: Interpreter? = null

        try {
            val options = Interpreter.Options()
            delegate = device.configure(options, numThreads, useFp16)
            interp = Interpreter(modelBuffer, options)
        } catch (e: Throwable) {
            android.util.Log.w("EasyML", "Delegate initialization with $device failed (${e.message}), falling back to CPU")
            delegate?.close()
            delegate = null
            val fallbackOptions = Interpreter.Options().apply {
                setUseXNNPACK(true)
                val availableCores = Runtime.getRuntime().availableProcessors()
                setNumThreads(minOf(availableCores, numThreads.coerceIn(1, 4)))
            }
            interp = Interpreter(modelBuffer, fallbackOptions)
        }

        this.gpuDelegate = delegate
        this.interpreter = interp
    }

    /**
     * Run inference with a single input/output (most common case).
     * Thread-safe via synchronized block.
     */
    fun run(input: ByteBuffer, output: ByteBuffer) {
        synchronized(lock) {
            interpreter.run(input, output)
        }
    }

    /**
     * Run inference with a single input and single output array.
     */
    fun run(input: Any, output: Any) {
        synchronized(lock) {
            interpreter.run(input, output)
        }
    }

    /**
     * Run inference with multiple inputs/outputs.
     * Use for models with multiple output heads.
     */
    fun runMultiOutput(inputs: Array<Any>, outputs: MutableMap<Int, Any>) {
        synchronized(lock) {
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        }
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
