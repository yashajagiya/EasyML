# 🚀 EasyML — The Complete Developer Guide & Architecture Reference (The EasyML Bible)

[![JitPack](https://img.shields.io/badge/JitPack-v1.6.1-brightgreen.svg)](https://jitpack.io/#yashajagiya/easyml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Min API](https://img.shields.io/badge/Min%20API-24%2B-brightgreen.svg)](https://developer.android.com/about/dashboards)
[![Kotlin](https://img.shields.io/badge/Kotlin-100%25%20Pure%20Kotlin-7F52FF.svg)](https://kotlinlang.org)
[![KMP Ready](https://img.shields.io/badge/Architecture-KMP%20Ready-F88909.svg)](https://kotlinlang.org/docs/multiplatform.html)
[![Compose](https://img.shields.io/badge/Jetpack%20Compose-Ready-4285F4.svg)](https://developer.android.com/jetpack/compose)

**EasyML** is an ultra-fast, 100% pure Kotlin, hardware-accelerated Machine Learning SDK engineered specifically for **Android**, **Jetpack Compose**, and future **Kotlin Multiplatform (KMP)** workflows.

It provides a unified, production-grade foundation for real-time **YOLO Object Detection (YOLO26, YOLO11, YOLOv8, YOLOv5)**, **Image Classification**, and **Custom TFLite Models** with zero boilerplate, multi-tier hardware acceleration (**GPU FP16 $\to$ NNAPI $\to$ Multi-Core CPU XNNPACK**), dynamic per-class color palettes, class-aware Non-Maximum Suppression (NMS), automatic INT8/UINT8 dequantization, and microsecond-level telemetry.

---

## 📑 Table of Contents

1. [🌟 Architectural Highlights & Core Pillars](#-architectural-highlights--core-pillars)
2. [📦 Complete Installation & All Gradle Configurations](#-complete-installation--all-gradle-configurations)
   - [Method 1: Version Catalog (`libs.versions.toml`) + Kotlin DSL](#method-1-version-catalog-libsversionstoml--kotlin-dsl-recommended)
   - [Method 2: Kotlin DSL Direct (`build.gradle.kts`)](#method-2-kotlin-dsl-direct-buildgradlekts)
   - [Method 3: Groovy DSL (`build.gradle`)](#method-3-groovy-dsl-buildgradle)
   - [JitPack Repository Setup (`settings.gradle.kts` vs Root `build.gradle`)](#jitpack-repository-setup)
   - [Manifest & Hardware Features](#manifest--hardware-features)
   - [Packaging Options (`noCompress`) for Zero-Copy Memory Mapping](#packaging-options-nocompress-for-zero-copy-memory-mapping)
   - [ProGuard / R8 Rules](#proguard--r8-rules)
3. [⚡ Direct Implementation (The 5-Line Quick Start)](#-direct-implementation-the-5-line-quick-start)
   - [Full Copy-Pasteable Activity](#full-copy-pasteable-activity)
   - [Line-by-Line Technical Breakdown](#line-by-line-technical-breakdown)
4. [🎨 Custom Implementations: The Deep-Dive Bible](#-custom-implementations-the-deep-dive-bible)
   - [1. Dynamic Styling Engine & Custom Canvas Overlays](#1-dynamic-styling-engine--custom-canvas-overlays)
   - [2. Manual Image Inference (Bitmaps, Gallery & Network)](#2-manual-image-inference-bitmaps-gallery--network)
   - [3. Zero-Allocation High-Throughput Memory Recycling](#3-zero-allocation-high-throughput-memory-recycling)
   - [4. Non-Blocking Coroutines (`detectAsync` & `loadDetectorAsync`)](#4-non-blocking-coroutines-detectasync--loaddetectorasync)
   - [5. Hardware Acceleration Tuning (GPU, NNAPI, CPU)](#5-hardware-acceleration-tuning-gpu-nnapi-cpu)
   - [6. Deep Dive into Model Decoders (YOLO26, YOLO11, YOLOv8, YOLOv5)](#6-deep-dive-into-model-decoders-yolo26-yolo11-yolov8-yolov5)
   - [7. Building a Custom Detection Decoder](#7-building-a-custom-detection-decoder)
   - [8. INT8 & UINT8 Quantization Dequantization](#8-int8--uint8-quantization-dequantization)
   - [9. Temporal Bounding Box Smoothing (`DetectionSmoother`)](#9-temporal-bounding-box-smoothing-detectionsmoother)
   - [10. Microsecond Telemetry & Profiling (`InferenceMetrics`)](#10-microsecond-telemetry--profiling-inferencemetrics)
   - [11. Image Classification Pipeline (`ImageClassifier`)](#11-image-classification-pipeline-imageclassifier)
   - [12. Raw Arbitrary Model Execution (`RawRunner`)](#12-raw-arbitrary-model-execution-rawrunner)
   - [13. JSON Serialization & Remote Streaming (`kotlinx.serialization`)](#13-json-serialization--remote-streaming-kotlinxserialization)
5. [🍳 Production Recipes & End-to-End Real-World Samples](#-production-recipes--end-to-end-real-world-samples)
   - [Recipe 1: Full-Featured Security Camera with Category Filtering](#recipe-1-full-featured-security-camera-with-category-filtering)
   - [Recipe 2: Gallery Image Analyzer in MVVM / ViewModel](#recipe-2-gallery-image-analyzer-in-mvvm--viewmodel)
   - [Recipe 3: Hazard Alert Sentry with Audio / Haptic Feedback & JSON Streaming](#recipe-3-hazard-alert-sentry-with-audio--haptic-feedback--json-streaming)
6. [🌐 100% Pure Kotlin & KMP Architecture](#-100-pure-kotlin--kmp-architecture)
7. [🛠️ Exhaustive API Reference](#️-exhaustive-api-reference)
8. [❓ FAQ, Troubleshooting & Performance Gotchas](#-faq-troubleshooting--performance-gotchas)
9. [📄 License](#-license)

---

## 🌟 Architectural Highlights & Core Pillars

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Jetpack Compose UI Layer                          │
│     <EasyMLCameraView />  •  <DetectionOverlay />  •  Custom Canvas     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Frame Lifecycle & Results State
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           EasyMLAnalyzer                                │
│   - Backpressure handling (STRATEGY_KEEP_ONLY_LATEST)                   │
│   - CameraX ISP Hardware Downscaling (ResolutionSelector 4:3)           │
│   - Opaque, upright frame acquisition (ImageUtils SIMD)                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Direct Invertible Matrix [Bitmap]
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ObjectDetector                                │
│   - Aspect-ratio letterboxing & padding extraction                      │
│   - Cache-friendly sequential memory normalizer                         │
│   - Invertible coordinate mapping (drawMatrix.invert)                   │
│   - Optional Exponential Moving Average (DetectionSmoother)             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Pre-allocated Native ByteBuffers
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TFLiteEngine Execution                            │
│  ┌───────────────────┐    ┌────────────────────┐    ┌────────────────┐  │
│  │ Tier 1: GPU FP16  │ -> │   Tier 2: NNAPI    │ -> │ Tier 3: CPU    │  │
│  │ OpenCL/OpenGL ES  │    │ NPU / DSP / Driver │    │ Multi-Thread   │  │
│  └───────────────────┘    └────────────────────┘    └────────────────┘  │
│   - Auto INT8/UINT8 Dequantization [(val - zeroPoint) * scale]          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Raw Output Tensors
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DetectionDecoder                                │
│   - AutoDetectionDecoder (Heuristic tensor shape matching)              │
│   - Yolo26EndToEndDecoder ([1, 300, 6] embedded NMS)                    │
│   - YoloV8Decoder ([1, 84, 8400] transposed / dual-head)                │
│   - YoloV5Decoder ([1, 25200, 85] with objectness)                      │
│   - Class-Aware NMS (Independent suppression per class index)           │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **100% Pure Kotlin**: Contains zero Java files and uses standard `src/main/kotlin` directory structure, paving the way for straightforward Kotlin Multiplatform (KMP) adoption.
2. **Resilient 3-Tier Hardware Acceleration**:
   - **Tier 1: GPU Delegate (FP16)** for ultra-fast shader computation on NHWC models.
   - **Tier 2: NNAPI Hardware Acceleration** for dispatching NCHW models (e.g. PyTorch YOLO exports) to modern phone NPUs, DSPs, and GPU drivers at **30+ FPS**.
   - **Tier 3: Multi-Core CPU (XNNPACK)** utilizing all performance cores on modern octa-core processors with ARM NEON SIMD.
3. **Zero-Allocation Pipeline**: Pre-allocated direct native byte buffers and in-place list population methods eliminate garbage collection (GC) pauses during live 60 FPS video streams.
4. **Dynamic Visual Engine**: High-contrast 16+ color dynamic class palette, translucent box fills, and frosted dark pill badges with confidence indicators.

---

## 📦 Complete Installation & All Gradle Configurations

EasyML is distributed through [JitPack](https://jitpack.io/#yashajagiya/easyml).

### JitPack Repository Setup

#### Modern Android Setup (`settings.gradle.kts`):
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

#### Legacy Android Setup (Root `build.gradle` or Root `build.gradle.kts`):
```groovy
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

---

### Dependency Declaration Options

#### Method 1: Version Catalog (`libs.versions.toml`) + Kotlin DSL (Recommended)

In `gradle/libs.versions.toml`:
```toml
[versions]
easyml = "1.6.1"

[libraries]
easyml = { group = "com.github.yashajagiya", name = "easyml", version.ref = "easyml" }
```

In `app/build.gradle.kts`:
```kotlin
dependencies {
    implementation(libs.easyml)
}
```

#### Method 2: Kotlin DSL Direct (`build.gradle.kts`)

In `app/build.gradle.kts`:
```kotlin
dependencies {
    implementation("com.github.yashajagiya:easyml:1.6.1")
}
```

#### Method 3: Groovy DSL (`build.gradle`)

In `app/build.gradle`:
```groovy
dependencies {
    implementation 'com.github.yashajagiya:easyml:1.6.1'
}
```

---

### Manifest & Hardware Features

Add to `app/src/main/AndroidManifest.xml`:
```xml
<!-- Required for live camera inference -->
<uses-feature android:name="android.hardware.camera.any" />
<uses-permission android:name="android.permission.CAMERA" />

<!-- Optional: Required only if reading photos from gallery or external storage -->
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
```

---

### Packaging Options (`noCompress`) for Zero-Copy Memory Mapping

> [!IMPORTANT]
> By default, the Android Gradle Plugin compresses `.tflite` model files in your APK. When compressed, TensorFlow Lite cannot memory-map the model directly into memory and must decompress the entire model into RAM, causing memory bloat and initialization lag.

Disable compression for `.tflite` files:

#### In Kotlin DSL (`app/build.gradle.kts`):
```kotlin
android {
    ...
    androidResources {
        noCompress += "tflite"
    }
}
```

#### In Groovy DSL (`app/build.gradle`):
```groovy
android {
    ...
    aaptOptions {
        noCompress 'tflite'
    }
}
```

---

### ProGuard / R8 Rules

EasyML includes consumer ProGuard rules automatically inside its `.aar`. If your app uses custom serialized data classes with `kotlinx.serialization` or custom decoders, verify your `app/proguard-rules.pro` includes:

```proguard
-keepattributes *Annotation*,Signature
-keepclassmembers class * {
    @kotlinx.serialization.Serializable *;
}
-keep class org.tensorflow.lite.** { *; }
```

---

## ⚡ Direct Implementation (The 5-Line Quick Start)

Get live camera object detection running in your Jetpack Compose app in seconds.

### Full Copy-Pasteable Activity

```kotlin
package com.example.mldependcay

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import com.easyml.EasyML
import com.easyml.camera.EasyMLCameraView
import com.easyml.core.InferenceDevice
import com.easyml.core.LabelSource
import com.easyml.core.ModelSource

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Scaffold(modifier = Modifier.fillMaxSize()) { paddingValues ->
                DetectionScreen(modifier = Modifier.padding(paddingValues))
            }
        }
    }
}

@Composable
fun DetectionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current

    // 1. Initialize detector with model, labels, and hardware acceleration
    val detector = remember {
        EasyML.objectDetector(context) {
            model = ModelSource.Asset("yolon.tflite")        // Model in app/src/main/assets/
            labels = LabelSource.Asset("labels.txt")         // Labels in app/src/main/assets/
            confidenceThreshold = 0.40f                      // Filter false positives
            device = InferenceDevice.AUTO                    // GPU FP16 -> NNAPI -> CPU
        }
    }

    // 2. Ensure interpreter and delegate memory are released on disposal
    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    // 3. Render camera stream, dynamic bounding boxes, and real-time FPS
    EasyMLCameraView(
        detector = detector,
        modifier = modifier.fillMaxSize(),
        showOverlay = true,
        showFps = true
    )
}
```

### Line-by-Line Technical Breakdown

1. `EasyML.objectDetector(context) { ... }`: The DSL entrypoint. Loads the TFLite flatbuffer, inspects input/output shapes, selects the appropriate decoder (e.g. YOLOv8, YOLO26, or YOLOv5), and configures the hardware engine.
2. `model = ModelSource.Asset("yolon.tflite")`: Resolves the model from `assets/` using zero-copy memory mapping (`AssetFileDescriptor`).
3. `labels = LabelSource.Asset("labels.txt")`: Reads class labels (one per line) into an immutable array.
4. `device = InferenceDevice.AUTO`: Automatically probes hardware:
   - Tries **GPU Delegate with FP16 precision**.
   - If GPU fails (e.g. NCHW layout in `yolon.tflite`), seamlessly falls back to **NNAPI hardware acceleration** (30+ FPS on NPU/DSP).
   - If NNAPI is unavailable, falls back to **Multi-Core CPU with XNNPACK**.
5. `<EasyMLCameraView />`: Handles camera runtime permissions, CameraX preview, ISP downscaling, inference execution, and dynamic canvas rendering.

---

## 🎨 Custom Implementations: The Deep-Dive Bible

### 1. Dynamic Styling Engine & Custom Canvas Overlays

EasyML includes a dynamic styling engine with a **16+ vibrant color palette** (`DetectionColorPalette`), translucent box fills, and frosted dark pill badges.

```
┌────────────────────────────────────────────────────────┐
│  [Person • 94%]                                        │  <- Frosted pill badge
│ ┌────────────────────────────────────────────────────┐ │
│ │                                                    │ │
│ │           (12% Translucent Tinted Fill)            │ │  <- Bounding box with
│ │                                                    │ │     rounded corners
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

#### Category or Object-Specific Custom Coloring
Supply a custom lambda to colorize detections dynamically:

```kotlin
EasyMLCameraView(
    detector = detector,
    modifier = Modifier.fillMaxSize(),
    colorProvider = { detection ->
        when (detection.label) {
            "person" -> Color(0xFF00E5FF)       // Electric Cyan for humans
            "knife", "gun" -> Color(0xFFFF1744) // Vibrant Red for hazard objects
            "car", "truck" -> Color(0xFFFF9100) // Sunset Orange for vehicles
            "dog", "cat" -> Color(0xFFFFD600)   // Amber Gold for pets
            else -> Color(0xFF9E9E9E)           // Gray for everything else
        }
    },
    strokeWidth = 4.0f,
    cornerRadius = 10.0f,
    showFill = true // Subtle 12% translucent fill inside each bounding box
)
```

#### Standalone `<DetectionOverlay />`
If you manage your own CameraX pipeline, render bounding boxes on any custom canvas:

```kotlin
DetectionOverlay(
    detections = detections.toDetectionList(),
    imageWidth = imageWidth,
    imageHeight = imageHeight,
    modifier = Modifier.fillMaxSize(),
    boxColor = null, // null triggers dynamic per-class palette
    colorProvider = null,
    strokeWidth = 3.5f,
    cornerRadius = 8f,
    showLabels = true,
    showConfidence = true,
    showFill = true,
    isMirrored = (lensFacing == CameraSelector.LENS_FACING_FRONT)
)
```

---

### 2. Manual Image Inference (Bitmaps, Gallery & Network)

You can run object detection synchronously on any Android `Bitmap`:

```kotlin
val bitmap: Bitmap = BitmapFactory.decodeFile("/sdcard/DCIM/sample.jpg")

// Synchronous detection (runs on current thread)
val detections: List<Detection> = detector.detect(bitmap)

detections.forEach { detection ->
    val box = detection.boundingBox
    Log.d("EasyML", "Found ${detection.label} (${(detection.confidence * 100).toInt()}%) at [$box]")
}
```

---

### 3. Zero-Allocation High-Throughput Memory Recycling

In high-frame-rate video loops (30–60 FPS), creating new `List<Detection>` instances on every frame causes garbage collection (GC) pressure and micro-stutters. EasyML allows you to populate an existing, reusable `MutableList`:

```kotlin
class VideoFrameProcessor(private val detector: ObjectDetector) {
    // Pre-allocated destination list reused for the lifetime of the pipeline
    private val reusableDetections = ArrayList<Detection>(50)

    fun onFrameArrived(frameBitmap: Bitmap) {
        // Populates reusableDetections in-place with ZERO heap allocations
        detector.detect(frameBitmap, reusableDetections)

        for (i in 0 until reusableDetections.size) {
            val item = reusableDetections[i]
            // Process detection without triggering GC events
        }
    }
}
```

---

### 4. Non-Blocking Coroutines (`detectAsync` & `loadDetectorAsync`)

EasyML provides native Kotlin Coroutine support to keep your Android Main/UI thread 100% fluid.

#### Non-Blocking Inference in ViewModel:
```kotlin
class DetectionViewModel : ViewModel() {
    private var detector: ObjectDetector? = null
    val detections = MutableStateFlow<List<Detection>>(emptyList())
    val isReady = MutableStateFlow(false)

    fun initDetector(context: Context) {
        viewModelScope.launch {
            // Offloads model weight reading & flatbuffer allocation to Dispatchers.IO
            detector = EasyML.loadDetectorAsync(context) {
                model = ModelSource.Asset("yolo11n.tflite")
                labels = LabelSource.Asset("labels.txt")
                confidenceThreshold = 0.45f
            }
            isReady.value = true
        }
    }

    fun analyzePhoto(bitmap: Bitmap) {
        viewModelScope.launch {
            // Offloads matrix letterboxing, TFLite inference, and NMS to Dispatchers.Default
            val results = detector?.detectAsync(bitmap) ?: emptyList()
            detections.value = results
        }
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
    }
}
```

---

### 5. Hardware Acceleration Tuning (GPU, NNAPI, CPU)

EasyML implements a resilient **3-tier hardware acceleration cascade**:

```kotlin
val detector = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolov8n.tflite")
    labels = LabelSource.Asset("labels.txt")
    
    // Hardware strategy:
    // - AUTO (Default): GPU (FP16) -> NNAPI (NPU/DSP) -> CPU (XNNPACK)
    // - GPU: Attempts GPU; falls back to CPU if unsupported
    // - GPU_STRICT: Throws exception if GPU delegate fails
    // - NNAPI: Forces Android Neural Networks API hardware acceleration
    // - CPU: Pure multi-core CPU with XNNPACK SIMD
    device = InferenceDevice.AUTO
    
    numThreads = 6 // Target performance threads for CPU execution
    useFp16 = true // Precision loss allowed for 2x faster GPU computation
}
```

#### Why NNAPI Solves the 5.5 FPS Bottleneck on NCHW Models
- **The Issue**: Many PyTorch/Ultralytics exports produce TFLite models with **NCHW** layout (`[1, 3, 640, 640]`). TensorFlow Lite's GPU delegate strictly requires **NHWC** layout (`[1, 640, 640, 3]`), causing standard GPU delegates to fail and drop directly to CPU (where an 8.7 GFLOP model runs at only ~5.5 FPS).
- **The EasyML Solution**: When the GPU delegate fails, `InferenceDevice.AUTO` automatically dispatches the model to **NNAPI** (Android Neural Networks API). NNAPI accelerates NCHW models directly on the phone's NPU, DSP, or GPU driver, lifting performance from 5.5 FPS to **30+ FPS**.

---

### 6. Deep Dive into Model Decoders (YOLO26, YOLO11, YOLOv8, YOLOv5)

EasyML decouples tensor parsing from model execution via the `DetectionDecoder` interface.

```kotlin
// 1. YOLO26 End-to-End (NMS embedded inside model graph: [1, 300, 6])
val detectorE2E = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolo26_e2e.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = Yolo26EndToEndDecoder(coordinateFormat = CoordinateFormat.AUTO)
}

// 2. YOLO11 / YOLOv8 (Transposed raw tensors: [1, 84, 8400])
val detectorV8 = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolov8n.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = YoloV8Decoder(isTransposed = true, coordinateFormat = CoordinateFormat.AUTO)
    classAgnosticNms = false // Class-aware: does not suppress overlapping objects of different classes
}

// 3. YOLOv5 / YOLOv7 (With Objectness Score: [1, 25200, 85])
val detectorV5 = EasyML.objectDetector(context) {
    model = ModelSource.Asset("yolov5s.tflite")
    labels = LabelSource.Asset("labels.txt")
    decoder = YoloV5Decoder(hasObjectness = true, coordinateFormat = CoordinateFormat.AUTO)
}
```

---

### 7. Building a Custom Detection Decoder

Have a custom SSD, EfficientDet, or proprietary model? Implement `DetectionDecoder`:

```kotlin
class CustomSSDDecoder : DetectionDecoder {
    override val isNmsFree: Boolean = false

    override fun decode(
        outputBuffer: FloatArray,
        outputShape: IntArray,
        inputWidth: Int,
        inputHeight: Int,
        confidenceThreshold: Float,
        candidates: MutableList<DetectionCandidate>
    ): Int {
        var count = 0
        // Parse custom tensor coordinates and append to candidates
        // Each candidate: DetectionCandidate(left, top, right, bottom, confidence, labelIndex)
        return count
    }
}
```

---

### 8. INT8 & UINT8 Quantization Dequantization

EasyML natively executes INT8 and UINT8 quantized models with zero user intervention:
- Inspects tensor zero points and scale factors automatically.
- Dequantizes integer outputs into normalized coordinates and confidences using:
  $$\text{floatValue} = (\text{quantizedValue} - \text{zeroPoint}) \times \text{scale}$$
- Works seamlessly across all decoders and classification engines.

---

### 9. Temporal Bounding Box Smoothing (`DetectionSmoother`)

To eliminate rapid bounding box jitter caused by per-frame prediction noise on live video feeds, EasyML provides Exponential Moving Average (EMA) box smoothing:

```kotlin
// In EasyMLCameraView:
EasyMLCameraView(
    detector = detector,
    enableSmoothing = true,
    smoothingFactor = 0.35f // 0.0 = frozen, 1.0 = raw unsmoothed
)

// Or as a standalone utility:
val smoother = DetectionSmoother(smoothingFactor = 0.35f)
val smoothedDetections = smoother.update(rawDetections)
```

---

### 10. Microsecond Telemetry & Profiling (`InferenceMetrics`)

EasyML provides high-precision telemetry measuring every segment of the pipeline:

```kotlin
EasyMLCameraView(
    detector = detector,
    onInferenceMetrics = { metrics ->
        Log.d("EasyML", "Preprocess: %.2f ms, Model: %.2f ms, Postprocess: %.2f ms, Total: %.2f ms (%.1f FPS)"
            .format(metrics.preprocessMs, metrics.inferenceMs, metrics.postprocessMs, metrics.totalMs, metrics.fps))
    }
)
```

---

### 11. Image Classification Pipeline (`ImageClassifier`)

EasyML includes a specialized classification engine for models like MobileNet, EfficientNet, or ResNet:

```kotlin
val classifier = EasyML.imageClassifier(context) {
    model = ModelSource.Asset("mobilenet_v3.tflite")
    labels = LabelSource.Asset("imagenet_labels.txt")
    maxResults = 5
    confidenceThreshold = 0.10f
    device = InferenceDevice.AUTO
}

val classifications = classifier.classify(bitmap)
classifications.forEach { item ->
    println("${item.label}: ${(item.confidence * 100).toInt()}%")
}
classifier.close()
```

---

### 12. Raw Arbitrary Model Execution (`RawRunner`)

For embedding generators, pose estimators, or custom multi-head architectures:

```kotlin
val rawRunner = EasyML.rawRunner(context) {
    model = ModelSource.Asset("feature_extractor.tflite")
    device = InferenceDevice.AUTO
}

rawRunner.run(inputDirectByteBuffer, outputDirectByteBuffer)
rawRunner.close()
```

---

### 13. JSON Serialization & Remote Streaming (`kotlinx.serialization`)

Every core data structure in EasyML is annotated with `@Serializable`:

```kotlin
val detections: List<Detection> = detector.detect(bitmap)

// 1. Serialize detections to JSON string (WebSockets / REST API)
val json: String = detections.toJson()

// 2. Deserialize from JSON string
val restored: List<Detection> = Detection.fromJsonList(json)

// 3. Serialize performance metrics
val metricsJson: String = detector.lastMetrics.toJson()
```

---

## 🍳 Production Recipes & End-to-End Real-World Samples

### Recipe 1: Full-Featured Security Camera with Category Filtering

```kotlin
@Composable
fun SecurityCameraScreen() {
    val context = LocalContext.current
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_BACK) }
    var alertCount by remember { mutableIntStateOf(0) }

    val detector = remember {
        EasyML.objectDetector(context) {
            model = ModelSource.Asset("yolov8n.tflite")
            labels = LabelSource.Asset("labels.txt")
            confidenceThreshold = 0.45f
            device = InferenceDevice.AUTO
        }
    }

    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        EasyMLCameraView(
            detector = detector,
            modifier = Modifier.fillMaxSize(),
            lensFacing = lensFacing,
            showOverlay = true,
            showFps = true,
            showFill = true,
            colorProvider = { detection ->
                if (detection.label in listOf("person", "knife", "backpack")) {
                    Color(0xFFFF1744) // Bright red alert for security items
                } else {
                    Color(0xFF00E5FF) // Electric cyan for everything else
                }
            },
            onResults = { results ->
                val securityObjects = results.count { it.label in listOf("person", "knife") }
                alertCount = securityObjects
            }
        )

        // Switch camera button
        IconButton(
            onClick = {
                lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                    CameraSelector.LENS_FACING_FRONT
                } else {
                    CameraSelector.LENS_FACING_BACK
                }
            },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(24.dp)
                .background(Color(0xCC000000), CircleShape)
        ) {
            Icon(Icons.Default.Refresh, contentDescription = "Switch Camera", tint = Color.White)
        }
    }
}
```

---

### Recipe 2: Gallery Image Analyzer in MVVM / ViewModel

```kotlin
class ImageAnalyzerViewModel : ViewModel() {
    private var detector: ObjectDetector? = null
    val detections = MutableStateFlow<List<Detection>>(emptyList())
    val isAnalyzing = MutableStateFlow(false)

    fun initialize(context: Context) {
        viewModelScope.launch {
            detector = EasyML.loadDetectorAsync(context) {
                model = ModelSource.Asset("yolo11n.tflite")
                labels = LabelSource.Asset("labels.txt")
            }
        }
    }

    fun analyzeBitmap(bitmap: Bitmap) {
        viewModelScope.launch {
            isAnalyzing.value = true
            val results = detector?.detectAsync(bitmap) ?: emptyList()
            detections.value = results
            isAnalyzing.value = false
        }
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
    }
}
```

---

### Recipe 3: Hazard Alert Sentry with Audio / Haptic Feedback & JSON Streaming

```kotlin
fun processLiveDetections(detections: List<Detection>, webSocketSession: WebSocketSession?) {
    val hazards = detections.filter { it.label in listOf("fire", "smoke", "weapon") }
    
    if (hazards.isNotEmpty()) {
        // Trigger haptic or audio feedback
        triggerAlertSound()
        
        // Stream telemetry over WebSocket as JSON
        val alertJson = hazards.toJson()
        webSocketSession?.send(Frame.Text(alertJson))
    }
}
```

---

## 🌐 100% Pure Kotlin & KMP Architecture

EasyML is designed from the ground up for modern Kotlin engineering:
- **Zero Java Code**: Every component (`core`, `detection`, `classification`, `camera`, `raw`) is written in idiomatic Kotlin.
- **Directory Layout**: Uses standard `src/main/kotlin` source trees.
- **Coroutines Native**: Built on `Dispatchers.Default` and `Dispatchers.IO`.
- **KMP-Ready Types**: Separation between mathematical detection structures (`Detection`, `DetectionList`, `InferenceMetrics`) and platform rendering layers.

---

## 🛠️ Exhaustive API Reference

### `DetectorConfig` DSL Parameters

| Property | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `ModelSource` | **Required** | `ModelSource.Asset("...")`, `ExternalFile(File)`, or `Buffer(ByteBuffer)`. |
| `labels` | `LabelSource?` | `null` | `LabelSource.Asset("...")`, `JsonAsset("...")`, or `StringList(...)`. |
| `confidenceThreshold` | `Float` | `0.25f` | Minimum confidence score `[0.0, 1.0]`. |
| `iouThreshold` | `Float` | `0.45f` | IoU overlap threshold for Non-Maximum Suppression. |
| `maxResults` | `Int` | `50` | Maximum detections to output. |
| `classAgnosticNms` | `Boolean` | `false` | If `true`, overlapping boxes of different classes suppress each other. |
| `decoder` | `DetectionDecoder` | `AutoDetectionDecoder` | Architecture decoder (`Yolo26EndToEndDecoder`, `YoloV8Decoder`, `YoloV5Decoder`). |
| `device` | `InferenceDevice` | `AUTO` | Hardware acceleration target (`AUTO`, `GPU`, `GPU_STRICT`, `NNAPI`, `CPU`). |
| `numThreads` | `Int` | `4` | Target CPU worker thread count. |
| `useFp16` | `Boolean` | `true` | Allows half-precision floating-point acceleration on GPU/NNAPI. |
| `inputWidth` | `Int?` | `null` | Optional override for model input width. |
| `inputHeight` | `Int?` | `null` | Optional override for model input height. |

### `EasyMLCameraView` Composable Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `detector` | `ObjectDetector` | **Required** | The initialized detector instance. |
| `modifier` | `Modifier` | `Modifier` | Compose layout modifier. |
| `showOverlay` | `Boolean` | `true` | Toggles rendering of bounding boxes and pill badges. |
| `showFps` | `Boolean` | `true` | Toggles the real-time FPS and latency counter pill. |
| `showFill` | `Boolean` | `true` | Renders a subtle 12% translucent tinted background inside boxes. |
| `overlayColor` | `Color?` | `null` | Color override. When `null`, automatic dynamic per-class palette is used. |
| `colorProvider` | `((Detection) -> Color)?` | `null` | Custom lambda to map any detection to a specific color. |
| `strokeWidth` | `Float` | `3.5f` | Bounding box line thickness in pixels. |
| `cornerRadius` | `Float` | `8.0f` | Rounded corner radius for boxes and tags. |
| `targetFps` | `Int?` | `null` | Optional frame-rate throttle to conserve device battery. |
| `enableSmoothing`| `Boolean` | `true` | Applies EMA temporal bounding box stabilization. |
| `smoothingFactor`| `Float` | `0.35f` | EMA weight factor between `0.0` and `1.0`. |
| `lensFacing` | `Int` | `LENS_FACING_BACK` | `CameraSelector.LENS_FACING_BACK` or `LENS_FACING_FRONT`. |
| `onResults` | `((List<Detection>) -> Unit)?` | `null` | Callback triggered on every analyzed frame. |
| `onInferenceMetrics` | `((InferenceMetrics) -> Unit)?`| `null` | Telemetry callback with microsecond latency breakdown. |

---

## ❓ FAQ, Troubleshooting & Performance Gotchas

#### Q: My frame rate is stuck at ~5.5 FPS. How do I fix it?
> **A:** Make sure you are using `device = InferenceDevice.AUTO`. If your model has an NCHW input tensor (`[1, 3, 640, 640]`), TFLite's GPU delegate will fail because it strictly requires NHWC (`[1, 640, 640, 3]`). EasyML's `InferenceDevice.AUTO` automatically falls back to **NNAPI**, which accelerates NCHW models at **30+ FPS** on hardware.

#### Q: Why are my bounding boxes not showing?
> **A:** 
> 1. Verify `confidenceThreshold` is not set too high (start with `0.25f` to `0.40f`).
> 2. Ensure your model export outputs either normalized coordinates `[0.0, 1.0]` or standard pixel coordinates `[0, 640]`. EasyML's `CoordinateFormat.AUTO` detects this automatically.
> 3. Verify that your camera frame conversion is delivering an opaque image. In EasyML, this is handled automatically via `ImageUtils.imageProxyToBitmap`.

#### Q: Can two overlapping objects of different classes both be detected?
> **A:** Yes! EasyML uses **Class-Aware NMS** by default. A person holding a cell phone or wearing a backpack will detect both objects. If you want class-agnostic suppression, set `classAgnosticNms = true`.

#### Q: How can I export my YOLO model for maximum speed?
> **A:** In Ultralytics, export using standard TFLite format:
> ```bash
> yolo export model=yolov8n.pt format=tflite
> ```
> For embedded NMS models (YOLO26), export with End-to-End enabled.

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
