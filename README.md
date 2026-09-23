# 🚀 EasyML — Hardware-Accelerated TFLite & YOLO Engine for Android & Jetpack Compose

[![JitPack](https://img.shields.io/badge/JitPack-v1.6.0-brightgreen.svg)](https://jitpack.io/#yashajagiya/easyml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Min API](https://img.shields.io/badge/Min%20API-24%2B-brightgreen.svg)](https://developer.android.com/about/dashboards)
[![Kotlin](https://img.shields.io/badge/Kotlin-Coroutines%20Ready-orange.svg)](https://kotlinlang.org)
[![Compose](https://img.shields.io/badge/Jetpack%20Compose-Supported-4285F4.svg)](https://developer.android.com/jetpack/compose)

**EasyML** is an Android-first, hardware-accelerated Machine Learning SDK engineered specifically for **Jetpack Compose** and modern Kotlin coroutines. It delivers robust, real-time inference for **YOLO Object Detection (YOLO26, YOLO11, YOLOv8, YOLOv5)**, **Image Classification**, and **Custom TFLite Models** with minimal boilerplate, class-aware Non-Maximum Suppression (NMS), automatic INT8/UINT8 dequantization, and microsecond-level telemetry.

---

## 📌 Table of Contents
- [✨ What's New in v1.6.0](#-whats-new-in-v160)
- [🧩 Feature & Model Support Matrix](#-feature--model-support-matrix)
- [💡 Why EasyML?](#-why-easyml)
- [⚡ Performance Architecture & Coroutines](#-performance-architecture--coroutines)
- [📦 Installation & Gradle Setup](#-installation--gradle-setup)
- [🚀 Quick Start: Jetpack Compose Camera](#-quick-start-jetpack-compose-camera)
- [📖 Deep-Dive Guides & Examples](#-deep-dive-guides--examples)
  - [1. YOLO Detection with Non-Blocking Coroutines (`detectAsync`)](#1-yolo-detection-with-non-blocking-coroutines-detectasync)
  - [2. Zero-Allocation High-Throughput Detection](#2-zero-allocation-high-throughput-detection)
  - [3. Asynchronous Model Loading (`loadDetectorAsync`)](#3-asynchronous-model-loading-loaddetectorasync)
  - [4. YOLO26 End-to-End (`[1, 300, 6]`) vs Raw YOLO Tensors](#4-yolo26-end-to-end-1-300-6-vs-raw-yolo-tensors)
  - [5. Microsecond Profiling via `InferenceMetrics`](#5-microsecond-profiling-via-inferencemetrics)
  - [6. Temporal Bounding Box Smoothing (`DetectionSmoother`)](#6-temporal-bounding-box-smoothing-detectionsmoother)
  - [7. Image Classification & Custom TFLite Models](#7-image-classification--custom-tflite-models)
  - [8. JSON Serialization & Export (`kotlinx.serialization`)](#8-json-serialization--export-kotlinxserialization)
- [🛠️ API Reference](#️-api-reference)
  - [DetectorConfig DSL](#detectorconfig-dsl)
  - [EasyMLCameraView Composable](#easymlcameraview-composable)
  - [InferenceDevice Options](#inferencedevice-options)
- [🔬 How EasyML Works Under the Hood](#-how-easyml-works-under-the-hood)
- [❓ Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [📄 License](#-license)

---

## ✨ What's New in v1.6.0

- 📦 **Official `kotlinx.serialization` Support**:
  - `Detection`, `DetectionList`, `Classification`, and `InferenceMetrics` are now `@Serializable`.
  - Added dedicated `RectFSerializer` for Android framework `android.graphics.RectF` bounding box coordinates.
  - Added convenient `.toJson()` and `.fromJson(...)` extension methods for direct export over WebSockets, REST APIs, or local disk.
  - Added `LabelSource.JsonAsset` and `LabelSource.JsonString` supporting both JSON arrays `["cat", "dog"]` and index-mapped objects `{"0": "cat", "1": "dog"}`.
- 🛠️ **Modern CameraX & TFLite API Cleanups**:
  - Migrated CameraX frame downscaling from deprecated `setTargetResolution` to modern `ResolutionSelector` & `ResolutionStrategy`.
  - Migrated GPU acceleration from deprecated `GpuDelegate.Options` to canonical `GpuDelegateFactory.Options`.
  - Cleaned up inspection warnings, redundant qualifiers, and spellchecker false positives.

- 🎯 **Class-Aware Non-Maximum Suppression (NMS)**: By default, overlapping boxes are suppressed only within the same class (e.g. a "dog" and a "leash" overlapping will no longer erase each other). Class-agnostic mode can be toggled via `classAgnosticNms = true`.
- ⚡ **YOLO26 Dual-Head & End-to-End Support**:
  - Native decoder for **YOLO26 End-to-End** (`[1, 300, 6]`) models with embedded NMS.
  - Native decoder for standard raw YOLO formats (`[1, 4+C, N]` and `[1, N, 4+C]`).
- 💎 **Pluggable `DetectionDecoder` Interface**: Decoupled tensor parsing from model execution. Choose `AutoDetectionDecoder`, `Yolo26EndToEndDecoder`, `YoloV8Decoder`, `YoloV5Decoder`, or supply your own custom decoder.
- 🚀 **Full Kotlin Coroutines Non-Blocking API**:
  - `suspend fun detectAsync(bitmap: Bitmap)`: Offloads preprocessing, inference, and postprocessing to `Dispatchers.Default` without blocking the main Android thread.
  - `suspend fun loadDetectorAsync(context, config)`: Moves flatbuffer parsing and weight allocation to `Dispatchers.IO`.
- 🔢 **Full INT8 / UINT8 Quantization Dequantization**: Automatically inspects tensor zero points and scale factors to dequantize integer outputs `(val - zeroPoint) * scale` into normalized probabilities and coordinates.
- ⏱️ **Microsecond Telemetry via `InferenceMetrics`**: Precise timing for preprocessing, native TFLite inference, and postprocessing via `System.nanoTime()`.
- 🌊 **Decoupled Temporal Smoothing (`DetectionSmoother`)**: Jitter-free bounding box rendering via Exponential Moving Average (EMA) box tracking without contaminating the stateless core detector.
- 📐 **Rectangular Model Dimensions**: Explicit `inputWidth` and `inputHeight` support (e.g., 384x640, 480x640).

---

## 🧩 Feature & Model Support Matrix

| Model Architecture | Output Tensor Shape | Built-in Decoder | Float32 | INT8 / UINT8 Quantized | Class-Aware NMS | Tested / Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO26 (End-to-End)** | `[1, 300, 6]` | `Yolo26EndToEndDecoder` | ✅ | ✅ (Auto-Dequant) | N/A (Embedded) | ✅ Production Ready |
| **YOLO26 (Raw Dual-Head)** | `[1, 4+C, N]` | `YoloV8Decoder` | ✅ | ✅ (Auto-Dequant) | ✅ Class-Aware | ✅ Production Ready |
| **YOLO11 / YOLOv8** | `[1, 4+C, 8400]` or transposed | `YoloV8Decoder` | ✅ | ✅ (Auto-Dequant) | ✅ (Optional Agnostic) | ✅ Production Ready |
| **YOLOv5 / YOLOv7** | `[1, 25200, 5+C]` (with objectness) | `YoloV5Decoder` | ✅ | ✅ (Auto-Dequant) | ✅ Class-Aware | ✅ Production Ready |
| **SSD / MobileNet-SSD** | Custom multi-tensor | `DetectionDecoder` (Custom) | ✅ | ✅ | Custom / Built-in | ✅ Extensible |
| **Image Classification** | `[1, NumClasses]` | `ImageClassifier` | ✅ | ✅ (Auto-Dequant) | N/A | ✅ Production Ready |
| **Custom Raw Models** | Arbitrary Tensors | `RawRunner` | ✅ | ✅ | Custom | ✅ Full Control |

---

## 💡 Why EasyML?

Integrating TensorFlow Lite directly on Android usually requires writing hundreds of lines of fragile boilerplate:
1. Converting `ImageProxy` / `YUV_420_888` camera frames into normalized inputs.
2. Allocating intermediate arrays and objects on every frame, generating garbage collector pressure and frame drops.
3. Handling multi-threading manually across Android UI and background threads.
4. Writing custom NMS algorithms that frequently contain multi-class overlap bugs.
5. Managing GPU delegate fallbacks when devices lack supported OpenCL/OpenGL ES drivers.

EasyML provides a clean, reactive architecture with safe GPU-to-CPU fallback, reusable memory buffers, coroutine-native dispatch, and seamless Jetpack Compose bindings.

---

## ⚡ Performance Architecture & Coroutines

EasyML is designed for high-frame-rate Android pipelines:

1. **Sequential Memory Scanning**: Box coordinates and class scores are processed sequentially in cache-friendly passes, maximizing L1/L2 CPU cache hits.
2. **Zero-Allocation Result Recycling**: An optional `detect(bitmap, outResults)` overload enables recycling an existing `MutableList<Detection>` across frames, avoiding per-frame list allocations.
3. **Coroutine Worker Pool Dispatch**: Call `detectAsync(bitmap)` from your ViewModel or Compose coroutine scope. It dispatches CPU-intensive matrix preprocessing and NMS onto `Dispatchers.Default` effortlessly.
4. **Pre-allocated Direct Native Buffers**: Camera and model input byte buffers are pre-allocated once during initialization.

---

## 📦 Installation & Gradle Setup

### Step 1: Add JitPack Repository

In your root `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

### Step 2: Add Dependency

In your `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.github.yashajagiya:easyml:1.5.0")
}
```

Or via Version Catalog (`gradle/libs.versions.toml`):

```toml
[versions]
easyml = "1.5.0"

[libraries]
easyml = { group = "com.github.yashajagiya", name = "easyml", version.ref = "easyml" }
```

### Step 3: Prevent Model File Compression

In `app/build.gradle.kts`, configure `androidResources` to prevent Android from compressing `.tflite` files in the APK, enabling zero-copy memory mapping:

```kotlin
android {
    ...
    androidResources {
        noCompress += "tflite"
    }
}
```

---

## 🚀 Quick Start: Jetpack Compose Camera

Add camera permission to `AndroidManifest.xml`:

```xml
<uses-feature android:name="android.hardware.camera.any" />
<uses-permission android:name="android.permission.CAMERA" />
```

Display real-time camera inference in Jetpack Compose:

```kotlin
@Composable
fun ObjectDetectionScreen() {
    val context = LocalContext.current
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }

    // 1. Initialize detector (or use EasyML.loadDetectorAsync inside a LaunchedEffect)
    val detector = remember {
        EasyML.objectDetector(context) {
            model = ModelSource.Asset("yolo26n.tflite")
            labels = LabelSource.Asset("labels.txt")
            confidenceThreshold = 0.40f
            iouThreshold = 0.45f
            device = InferenceDevice.AUTO // Probes GPU FP16; falls back to CPU XNNPACK
        }
    }

    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    // 2. Camera Preview with overlay and FPS counter
    EasyMLCameraView(
        detector = detector,
        modifier = Modifier.fillMaxSize(),
        showOverlay = true,
        showFps = true,
        enableSmoothing = true,       // Stabilizes bounding box jitter
        smoothingFactor = 0.35f,
        onResults = { results ->
            detections = results
        },
        onInferenceMetrics = { metrics ->
            // Telemetry: metrics.inferenceMs, metrics.totalMs
        }
    )
}
```

---

## 📖 Deep-Dive Guides & Examples

### 1. YOLO Detection with Non-Blocking Coroutines (`detectAsync`)

When running inference outside of CameraX (e.g. processing gallery photos, custom video streams, or ViewModel pipelines), use the non-blocking `detectAsync` suspend function to prevent UI stutters:

```kotlin
class DetectionViewModel : ViewModel() {
    private var detector: ObjectDetector? = null

    fun initialize(context: Context) {
        viewModelScope.launch {
            // Asynchronously parse model and allocate buffers on Dispatchers.IO
            detector = EasyML.loadDetectorAsync(context) {
                model = ModelSource.Asset("yolo11n.tflite")
                labels = LabelSource.Asset("coco_labels.txt")
                confidenceThreshold = 0.50f
            }
        }
    }

    fun processImage(bitmap: Bitmap) {
        viewModelScope.launch {
            // Dispatches preprocessing, TFLite inference, and NMS on Dispatchers.Default
            val results: List<Detection> = detector?.detectAsync(bitmap) ?: emptyList()
            _detections.value = results
        }
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
    }
}
```

---

### 2. Zero-Allocation High-Throughput Detection

For maximum memory efficiency in high-frequency video processing, reuse a single `MutableList<Detection>` across frames:

```kotlin
val reusableList = mutableListOf<Detection>()

fun onFrameArrived(frameBitmap: Bitmap) {
    // Zero heap allocation: populates reusableList in place
    detector.detect(frameBitmap, reusableList)

    for (i in 0 until reusableList.size) {
        val detection = reusableList[i]
        // Process detection without triggering GC events
    }
}
```

---

### 3. Asynchronous Model Loading (`loadDetectorAsync`)

Model loading can take 100–300ms depending on model size and device storage speed. `loadDetectorAsync` loads the weights on `Dispatchers.IO`:

```kotlin
@Composable
fun AsyncDetectorScreen() {
    val context = LocalContext.current
    var detector by remember { mutableStateOf<ObjectDetector?>(null) }
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        detector = EasyML.loadDetectorAsync(context) {
            model = ModelSource.Asset("yolo26n.tflite")
            labels = LabelSource.Asset("labels.txt")
        }
        isLoading = false
    }

    if (isLoading) {
        CircularProgressIndicator()
    } else {
        detector?.let { safeDetector ->
            EasyMLCameraView(detector = safeDetector)
        }
    }
}
```

---

### 4. YOLO26 End-to-End (`[1, 300, 6]`) vs Raw YOLO Tensors

EasyML auto-detects model tensor shapes by default via `AutoDetectionDecoder`. You can also configure the exact decoder explicitly:

```kotlin
// 1. YOLO26 End-to-End (NMS built into model graph: [1, 300, 6])
val e2eDetector = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolo26_e2e.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = Yolo26EndToEndDecoder
}

// 2. Standard YOLOv8 / YOLO11 / YOLO26 Raw Dual-Head ([1, 4+C, N])
val rawDetector = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolov8n.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = YoloV8Decoder(isTransposed = true)
    classAgnosticNms = false // Class-aware NMS preserves overlapping objects of different classes
}

// 3. YOLOv5 / YOLOv7 with Objectness Score ([1, 25200, 5+C])
val v5Detector = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolov5s.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = YoloV5Decoder
}
```

---

### 5. Microsecond Profiling via `InferenceMetrics`

EasyML instruments every phase of the detection pipeline using `System.nanoTime()`:

```kotlin
val detections = detector.detect(bitmap)
detector.lastInferenceMetrics?.let { metrics ->
    println("Preprocess  : ${metrics.preprocessMs} ms")
    println("TFLite Exec : ${metrics.inferenceMs} ms")
    println("NMS Postproc: ${metrics.postprocessMs} ms")
    println("Total Time  : ${metrics.totalMs} ms (${metrics.fps.toInt()} FPS)")
}
```

In `<EasyMLCameraView />`, access real-time metrics using `onInferenceMetrics`:

```kotlin
EasyMLCameraView(
    detector = detector,
    onInferenceMetrics = { metrics ->
        Log.d("EasyML", "Inference: ${metrics.inferenceMs} ms, Post: ${metrics.postprocessMs} ms")
    }
)
```

---

### 6. Temporal Bounding Box Smoothing (`DetectionSmoother`)

To prevent bounding box jitter caused by per-frame prediction noise on live camera feeds, use the standalone `DetectionSmoother`:

```kotlin
val smoother = DetectionSmoother(smoothingFactor = 0.35f, iouThreshold = 0.50f)

fun onNewFrame(rawDetections: List<Detection>): List<Detection> {
    return smoother.smooth(rawDetections)
}

// Clear temporal history when switching scenes or cameras
smoother.reset()
```

---

### 7. Image Classification & Custom TFLite Models

#### Image Classification:
```kotlin
val classifier = EasyML.classifier(context) {
    model = ModelSource.Asset("mobilenet_v3.tflite")
    labels = LabelSource.Asset("imagenet_labels.txt")
    maxResults = 5
    confidenceThreshold = 0.05f
    device = InferenceDevice.AUTO
}

val results = classifier.classify(bitmap)
results.forEach { println("${it.label}: ${(it.confidence * 100).toInt()}%") }
classifier.close()
```

#### Raw Custom Model Execution:
```kotlin
val runner = EasyML.raw(context) {
    model = ModelSource.Asset("custom_model.tflite")
    device = InferenceDevice.AUTO
}

runner.run(inputDirectByteBuffer, outputDirectByteBuffer)
runner.close()
```

---

### 8. JSON Serialization & Export (`kotlinx.serialization`)

EasyML includes built-in `kotlinx.serialization` support for exporting inference results, logging telemetry, and loading JSON labels.

#### Export Detections to JSON (REST API / WebSocket / Disk):
```kotlin
val detections: List<Detection> = detector.detect(bitmap)

// Serialize List<Detection> or single Detection
val jsonString = detections.toJson()

// Or serialize DetectionList
val detectionList = detections.toDetectionList()
val jsonString = detectionList.toJson()

// Deserialize back to Detection objects
val restoredDetections = Detection.fromJsonList(jsonString)
```

#### Export Inference Latency Metrics:
```kotlin
detector.lastInferenceMetrics?.let { metrics ->
    val metricsJson = metrics.toJson()
    // Send to Datadog, Firebase, or custom analytics
}
```

#### Load JSON Label Files:
```kotlin
// Load labels from JSON array: ["cat", "dog", "car"]
// or index-mapped JSON object: {"0": "cat", "1": "dog"}
EasyML.detector(context) {
    model = ModelSource.Asset("yolov8n.tflite")
    labels = LabelSource.JsonAsset("coco_classes.json")
}
```

---

## 🛠️ API Reference

### `DetectorConfig` DSL

| Property | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `ModelSource` | **Required** | Model source (`Asset`, `ExternalFile`, or `Buffer`). |
| `labels` | `LabelSource?` | `null` | Label source mapping class IDs to human-readable names. |
| `confidenceThreshold` | `Float` | `0.25f` | Minimum score threshold `[0.0, 1.0]`. |
| `iouThreshold` | `Float` | `0.45f` | Intersection-over-Union threshold for NMS. |
| `maxResults` | `Int` | `50` | Maximum detections returned. |
| `classAgnosticNms` | `Boolean` | `false` | When `false`, NMS only suppresses boxes within the same class. |
| `decoder` | `DetectionDecoder` | `AutoDetectionDecoder` | Output parser (`AutoDetectionDecoder`, `Yolo26EndToEndDecoder`, etc.). |
| `device` | `InferenceDevice` | `AUTO` | Acceleration strategy (`AUTO`, `GPU_STRICT`, `CPU`, `NNAPI`). |
| `numThreads` | `Int` | `4` | Number of CPU worker threads. |
| `inputWidth` | `Int?` | `null` | Optional override for model input width. |
| `inputHeight` | `Int?` | `null` | Optional override for model input height. |

---

### `EasyMLCameraView` Composable

```kotlin
@Composable
fun EasyMLCameraView(
    detector: ObjectDetector,
    modifier: Modifier = Modifier,
    showOverlay: Boolean = true,
    showFps: Boolean = true,
    enableSmoothing: Boolean = false,
    smoothingFactor: Float = 0.35f,
    lensFacing: Int = CameraSelector.LENS_FACING_BACK,
    overlayColor: Color = Color(0xFF00E676),
    onResults: ((List<Detection>) -> Unit)? = null,
    onInferenceMetrics: ((InferenceMetrics) -> Unit)? = null
)
```

---

### `InferenceDevice` Options

- `InferenceDevice.AUTO` / `InferenceDevice.GPU_OR_CPU`: **Recommended**. Probes device OpenCL/OpenGL ES capabilities. Attempts GPU Delegate with FP16 precision, safely falling back to CPU XNNPACK SIMD if initialization fails.
- `InferenceDevice.GPU_STRICT`: Forces GPU execution and throws an exception if unsupported.
- `InferenceDevice.CPU`: Executes on CPU using ARM NEON multi-threaded XNNPACK.
- `InferenceDevice.NNAPI`: Hardware acceleration via Android Neural Networks API.

---

## 🔬 How EasyML Works Under the Hood

```
┌─────────────────────────────────────────────────────────────┐
│                      Jetpack Compose                        │
│                   <EasyMLCameraView />                      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Live Camera Frame (ImageProxy / RGBA)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      EasyMLAnalyzer                         │
│  - Non-blocking frame throttle (STRATEGY_KEEP_ONLY_LATEST)  │
│  - Optional Temporal Box Smoother (DetectionSmoother)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Direct Pre-allocated ByteBuffer
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   TFLiteEngine Execution                    │
│  - Hardware Accelerated (GPU Delegate FP16 / CPU XNNPACK)   │
│  - Auto INT8/UINT8 Dequantization [(val - zeroPoint) * scale]
└──────────────────────────────┬──────────────────────────────┘
                               │ Raw Output Tensor
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      DetectionDecoder                       │
│  - AutoDetectionDecoder / Yolo26EndToEnd / YoloV8 / YoloV5  │
│  - Class-Aware NMS (Class-isolated candidate suppression)   │
│  - Microsecond Telemetry (InferenceMetrics via nanoTime)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ List<Detection>
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      DetectionOverlay                       │
│  - Aspect-ratio matched canvas bounding box rendering       │
└─────────────────────────────────────────────────────────────┘
```

---

## ❓ Frequently Asked Questions (FAQ)

#### Q: How does EasyML handle different YOLO versions?
> **A:** EasyML includes dedicated decoders for YOLOv5 (with objectness scores), YOLOv8 / YOLO11 (separate class scores and coordinates), and YOLO26 End-to-End (`[1, 300, 6]`). By default, `AutoDetectionDecoder` inspects the output tensor shape and delegates to the appropriate decoder automatically.

#### Q: Does EasyML suppress overlapping objects of different classes?
> **A:** No. In v1.5.0, Class-Aware NMS is enabled by default. A person and a backpack overlapping in the frame will each be preserved. If you require class-agnostic suppression, set `classAgnosticNms = true` in your `DetectorConfig`.

#### Q: Can I run INT8 or UINT8 quantized models?
> **A:** Yes. EasyML automatically inspects the model tensor's quantization parameters (quantization scale and zero point). Integer values are automatically converted to normalized floating-point coordinates and probabilities.

---

## 📄 License

```text
Copyright 2026 EasyML Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
