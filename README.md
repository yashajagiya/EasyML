# 🚀 EasyML — The Complete Guide & Architecture Reference (The EasyML Bible)

[![JitPack](https://img.shields.io/badge/JitPack-v1.6.1-brightgreen.svg)](https://jitpack.io/#yashajagiya/easyml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Min API](https://img.shields.io/badge/Min%20API-24%2B-brightgreen.svg)](https://developer.android.com/about/dashboards)
[![Kotlin](https://img.shields.io/badge/Kotlin-100%25%20Pure%20Kotlin-7F52FF.svg)](https://kotlinlang.org)
[![KMP Ready](https://img.shields.io/badge/Architecture-KMP%20Ready-F88909.svg)](https://kotlinlang.org/docs/multiplatform.html)
[![Compose](https://img.shields.io/badge/Jetpack%20Compose-Ready-4285F4.svg)](https://developer.android.com/jetpack/compose)

**EasyML** is an ultra-fast, 100% pure Kotlin, hardware-accelerated Machine Learning SDK engineered specifically for Android, Jetpack Compose, and future Kotlin Multiplatform (KMP) workflows.

It delivers real-time inference for **YOLO Object Detection (YOLO26, YOLO11, YOLOv8, YOLOv5)**, **Image Classification**, and **Custom TFLite Models** with zero boilerplate, multi-tier hardware acceleration (**GPU FP16 $\to$ NNAPI $\to$ Multi-Core CPU XNNPACK**), dynamic per-class color palettes, class-aware Non-Maximum Suppression (NMS), automatic INT8/UINT8 dequantization, and microsecond-level telemetry.

---

## 📑 Table of Contents

1. [🌟 Core Highlights & Architectural Pillars](#-core-highlights--architectural-pillars)
2. [📦 Installation & All Gradle Configurations](#-installation--all-gradle-configurations)
   - [Method 1: Kotlin DSL with Version Catalog (`libs.versions.toml`)](#method-1-kotlin-dsl-with-version-catalog-libsversionstoml)
   - [Method 2: Kotlin DSL (`build.gradle.kts`)](#method-2-kotlin-dsl-buildgradlekts)
   - [Method 3: Groovy DSL (`build.gradle`)](#method-3-groovy-dsl-buildgradle)
   - [JitPack Repository Setup](#jitpack-repository-setup)
   - [Manifest & Packaging Configuration (`noCompress`)](#manifest--packaging-configuration-nocompress)
   - [ProGuard / R8 Configuration](#proguard--r8-configuration)
3. [⚡ Direct Implementation (Quick Start in 5 Lines)](#-direct-implementation-quick-start-in-5-lines)
   - [Full Working Example](#full-working-example)
   - [Line-by-Line Breakdown](#line-by-line-breakdown)
4. [🎨 Custom Implementations (The Deep-Dive Bible)](#-custom-implementations-the-deep-dive-bible)
   - [1. Dynamic Per-Class Visuals & Custom Overlays](#1-dynamic-per-class-visuals--custom-overlays)
   - [2. Manual Image Detection (Bitmaps, Gallery, Network)](#2-manual-image-detection-bitmaps-gallery-network)
   - [3. Zero-Allocation High-Throughput Memory Recycling](#3-zero-allocation-high-throughput-memory-recycling)
   - [4. Non-Blocking Coroutines (`detectAsync` & `loadDetectorAsync`)](#4-non-blocking-coroutines-detectasync--loaddetectorasync)
   - [5. Hardware Acceleration Tuning (GPU, NNAPI, CPU)](#5-hardware-acceleration-tuning-gpu-nnapi-cpu)
   - [6. Model Decoders (YOLO26 End-to-End, YOLO11/v8, YOLOv5 & Custom)](#6-model-decoders-yolo26-end-to-end-yolo11v8-yolov5--custom)
   - [7. INT8 / UINT8 Quantization & Automatic Dequantization](#7-int8--uint8-quantization--automatic-dequantization)
   - [8. Temporal Bounding Box Smoothing (`DetectionSmoother`)](#8-temporal-bounding-box-smoothing-detectionsmoother)
   - [9. Microsecond Profiling via `InferenceMetrics`](#9-microsecond-profiling-via-inferencemetrics)
   - [10. Image Classification Pipeline](#10-image-classification-pipeline)
   - [11. Raw Arbitrary Model Runner](#11-raw-arbitrary-model-runner)
   - [12. JSON Serialization & Remote Streaming (`kotlinx.serialization`)](#12-json-serialization--remote-streaming-kotlinxserialization)
5. [🌐 100% Pure Kotlin & KMP Architecture](#-100-pure-kotlin--kmp-architecture)
6. [🛠️ Complete API Reference](#️-complete-api-reference)
7. [❓ FAQ & Troubleshooting](#-faq--troubleshooting)
8. [📄 License](#-license)

---

## 🌟 Core Highlights & Architectural Pillars

| Pillar | Capability |
| :--- | :--- |
| **100% Pure Kotlin** | Zero Java files in the library. Clean `src/main/kotlin` architecture designed for straightforward KMP migration. |
| **3-Tier Hardware Acceleration** | Automatic fallback: **GPU Delegate (FP16)** $\to$ **NNAPI (NPU/DSP/GPU)** $\to$ **Multi-Core CPU (XNNPACK)**. Accelerates NCHW models at 30+ FPS. |
| **All-in-One Compose View** | `<EasyMLCameraView />` encapsulates camera permissions, CameraX lifecycle, ISP downscaling, box smoothing, and dynamic rendering into one composable. |
| **Dynamic Styling Engine** | High-contrast 16+ color dynamic class palette, translucent bounding box fills, and frosted dark pill badges with confidence indicators. |
| **YOLO Multi-Architecture** | Native support for **YOLO26 End-to-End** (`[1, 300, 6]`), **YOLO11 / YOLOv8** (`[1, 84, 8400]`), and **YOLOv5 / YOLOv7** (`[1, 25200, 85]`). |
| **Zero-Allocation Pipeline** | Reusable `ByteBuffer` memory pools and in-place list population eliminate garbage collector pauses during live camera streams. |
| **First-Class Serialization** | Native `kotlinx.serialization` support for `Detection`, `DetectionList`, `Classification`, and `InferenceMetrics`. |

---

## 📦 Installation & All Gradle Configurations

EasyML is distributed via [JitPack](https://jitpack.io/#yashajagiya/easyml). Choose the Gradle configuration that matches your project setup.

### JitPack Repository Setup

#### Modern Android (`settings.gradle.kts`):
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

#### Legacy Android (`root build.gradle` or `root build.gradle.kts`):
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

### Dependency Declaration

#### Method 1: Kotlin DSL with Version Catalog (`libs.versions.toml`) — Recommended
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

#### Method 2: Kotlin DSL (`build.gradle.kts`)
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

### Manifest & Packaging Configuration (`noCompress`)

1. **Camera Permission**: Add to `app/src/main/AndroidManifest.xml`:
```xml
<uses-feature android:name="android.hardware.camera.any" />
<uses-permission android:name="android.permission.CAMERA" />
```

2. **Disable Model Compression**: TensorFlow Lite models must be uncompressed in the APK so they can be memory-mapped zero-copy directly from disk into hardware accelerators.

In `app/build.gradle.kts`:
```kotlin
android {
    ...
    androidResources {
        noCompress += "tflite"
    }
}
```
Or in Groovy `app/build.gradle`:
```groovy
android {
    ...
    aaptOptions {
        noCompress 'tflite'
    }
}
```

### ProGuard / R8 Configuration
EasyML includes consumer ProGuard rules automatically. If you write custom decoders or model classes with `kotlinx.serialization`, ensure R8 preserves serialized names:
```proguard
-keepattributes *Annotation*,Signature
-keepclassmembers class * {
    @kotlinx.serialization.Serializable *;
}
```

---

## ⚡ Direct Implementation (Quick Start in 5 Lines)

The fastest way to get real-time object detection running in your Android app is using `<EasyMLCameraView />`.

### Full Working Example

```kotlin
package com.example.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
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
        setContent {
            DetectionScreen()
        }
    }
}

@Composable
fun DetectionScreen() {
    val context = LocalContext.current

    // Line 1: Build the detector instance
    val detector = remember {
        EasyML.objectDetector(context) {
            model = ModelSource.Asset("yolov8n.tflite")        // Model in assets/
            labels = LabelSource.Asset("labels.txt")           // Labels in assets/
            confidenceThreshold = 0.40f                        // Filter false positives
            device = InferenceDevice.AUTO                      // GPU FP16 -> NNAPI -> CPU
        }
    }

    // Line 2: Ensure resources close when screen disposes
    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    // Line 3-5: Live camera preview + dynamic boxes + FPS badge
    EasyMLCameraView(
        detector = detector,
        modifier = Modifier.fillMaxSize(),
        showOverlay = true,
        showFps = true
    )
}
```

### Line-by-Line Breakdown

1. `EasyML.objectDetector(context) { ... }`: The DSL entrypoint. Automatically loads the TFLite flatbuffer, inspects input/output shapes, selects the appropriate decoder (e.g. YOLOv8, YOLO26, or YOLOv5), and configures the hardware engine.
2. `model = ModelSource.Asset("yolov8n.tflite")`: Resolves the model from the `app/src/main/assets/` directory using zero-copy memory mapping.
3. `labels = LabelSource.Asset("labels.txt")`: Loads human-readable names for classes (e.g. `person`, `car`, `dog`).
4. `device = InferenceDevice.AUTO`: Automatically probes hardware. It tries the GPU delegate with FP16 precision, then attempts **NNAPI** hardware acceleration, and finally falls back to multi-core CPU with XNNPACK SIMD.
5. `<EasyMLCameraView />`: Handles everything: requests camera permission, binds CameraX Preview and ImageAnalysis with hardware downscaling, runs real-time inference, and renders the dynamic overlay.

---

## 🎨 Custom Implementations (The Deep-Dive Bible)

### 1. Dynamic Per-Class Visuals & Custom Overlays

EasyML comes with an automatic **16+ vibrant neon color palette** (`DetectionColorPalette`) and floating frosted pill badges. You can customize every aspect of the visual rendering.

#### Automatic Dynamic Mode (Default)
When `overlayColor` is omitted or set to `null`, every class is assigned a distinct high-contrast color automatically:
```kotlin
EasyMLCameraView(
    detector = detector,
    modifier = Modifier.fillMaxSize(),
    showOverlay = true,
    showFill = true          // Subtle 12% translucent fill inside each bounding box
)
```

#### Custom Color Provider (Category or Object-Specific)
Supply a custom lambda to colorize detections dynamically:
```kotlin
EasyMLCameraView(
    detector = detector,
    modifier = Modifier.fillMaxSize(),
    colorProvider = { detection ->
        when (detection.label) {
            "person" -> Color(0xFF00E5FF)       // Electric Cyan for humans
            "knife", "gun" -> Color(0xFFFF1744) // Vibrant Red for hazard objects
            "cat", "dog" -> Color(0xFFFFD600)   // Gold for animals
            else -> Color(0xFF9E9E9E)           // Gray for everything else
        }
    },
    strokeWidth = 4.0f,
    cornerRadius = 10.0f,
    showFill = true
)
```

#### Standalone Compose Overlay Canvas
If you manage your own CameraX `PreviewView`, you can render the bounding boxes using `<DetectionOverlay />` independently:
```kotlin
DetectionOverlay(
    detections = detections.toDetectionList(),
    imageWidth = imageWidth,
    imageHeight = imageHeight,
    modifier = Modifier.fillMaxSize(),
    colorProvider = { det -> DetectionColorPalette.getColorForClass(det.labelIndex) },
    strokeWidth = 3.5f,
    cornerRadius = 8f,
    showLabels = true,
    showConfidence = true,
    showFill = true,
    isMirrored = isFrontCamera
)
```

---

### 2. Manual Image Detection (Bitmaps, Gallery, Network)

You don't need a camera to use EasyML. You can run object detection synchronously or asynchronously on any Android `Bitmap`:

```kotlin
val bitmap: Bitmap = BitmapFactory.decodeFile("/path/to/image.jpg")

// Synchronous execution (blocking)
val detections: List<Detection> = detector.detect(bitmap)

detections.forEach { detection ->
    println("Detected ${detection.label} (${(detection.confidence * 100).toInt()}%) at ${detection.boundingBox}")
}
```

---

### 3. Zero-Allocation High-Throughput Memory Recycling

In high-frame-rate video loops (30–60 FPS), creating new `List<Detection>` instances on every frame causes garbage collection (GC) pressure and micro-stutters. EasyML allows you to populate an existing, reusable `MutableList`:

```kotlin
class FrameProcessor(private val detector: ObjectDetector) {
    // Pre-allocated destination list reused for the lifetime of the pipeline
    private val reusableDetections = ArrayList<Detection>(50)

    fun onFrame(frameBitmap: Bitmap) {
        // Populates reusableDetections in-place with ZERO heap allocations
        detector.detect(frameBitmap, reusableDetections)

        for (i in 0 until reusableDetections.size) {
            val item = reusableDetections[i]
            // Process detection
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

    fun initDetector(context: Context) {
        viewModelScope.launch {
            // Offloads model weight reading & flatbuffer allocation to Dispatchers.IO
            detector = EasyML.loadDetectorAsync(context) {
                model = ModelSource.Asset("yolo11n.tflite")
                labels = LabelSource.Asset("labels.txt")
            }
        }
    }

    fun analyzePhoto(bitmap: Bitmap) {
        viewModelScope.launch {
            // Offloads matrix letterboxing, TFLite inference, and NMS to Dispatchers.Default
            val results = detector?.detectAsync(bitmap) ?: emptyList()
            detections.value = results
        }
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
    
    // Acceleration Options:
    // 1. InferenceDevice.AUTO (Recommended)
    //    Tier 1: GPU Delegate with FP16 precision
    //    Tier 2: NNAPI Hardware Acceleration (NPU/DSP/GPU)
    //    Tier 3: Multi-core CPU with XNNPACK SIMD
    device = InferenceDevice.AUTO
    
    numThreads = 6 // Target performance threads for CPU execution
    useFp16 = true // Precision loss allowed for 2x faster GPU computation
}
```

#### Why NNAPI Solves the 5.5 FPS Bottleneck on NCHW Models
- **The Issue**: Many PyTorch/Ultralytics exports produce TFLite models with **NCHW** layout (`[1, 3, 640, 640]`). TensorFlow Lite's GPU delegate strictly requires **NHWC** layout (`[1, 640, 640, 3]`), causing standard GPU delegates to fail and drop directly to CPU (where an 8.7 GFLOP model runs at only ~5.5 FPS).
- **The EasyML Solution**: When the GPU delegate fails, `InferenceDevice.AUTO` automatically dispatches the model to **NNAPI** (Android Neural Networks API). NNAPI accelerates NCHW models directly on the phone's NPU, DSP, or GPU driver, lifting performance from 5.5 FPS to **30+ FPS**.

---

### 6. Model Decoders (YOLO26 End-to-End, YOLO11/v8, YOLOv5 & Custom)

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

#### Writing a Custom Decoder
Have a custom SSD, MobileNet, or proprietary model? Implement `DetectionDecoder`:
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
        return count
    }
}
```

---

### 7. INT8 / UINT8 Quantization & Automatic Dequantization

EasyML natively executes INT8 and UINT8 quantized models with zero user intervention:
- Inspects tensor zero points and scale factors automatically.
- Dequantizes integer outputs into normalized coordinates and confidences using:
  $$\text{floatValue} = (\text{quantizedValue} - \text{zeroPoint}) \times \text{scale}$$
- Works seamlessly across all decoders and classification engines.

---

### 8. Temporal Bounding Box Smoothing (`DetectionSmoother`)

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

### 9. Microsecond Profiling via `InferenceMetrics`

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

### 10. Image Classification Pipeline

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

### 11. Raw Arbitrary Model Runner

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

### 12. JSON Serialization & Remote Streaming (`kotlinx.serialization`)

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

## 🌐 100% Pure Kotlin & KMP Architecture

EasyML is designed from the ground up for modern Kotlin engineering:
- **Zero Java Code**: Every component (`core`, `detection`, `classification`, `camera`, `raw`) is written in idiomatic Kotlin.
- **Directory Layout**: Uses standard `src/main/kotlin` source trees.
- **Coroutines Native**: Built on `Dispatchers.Default` and `Dispatchers.IO`.
- **KMP-Ready Types**: Separation between mathematical detection structures (`Detection`, `DetectionList`, `InferenceMetrics`) and platform rendering layers.

---

## 🛠️ Complete API Reference

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

## ❓ FAQ & Troubleshooting

#### Q: My frame rate is stuck at ~5.5 FPS. How do I fix it?
> **A:** Make sure you are using `device = InferenceDevice.AUTO`. If your model has an NCHW input tensor (`[1, 3, 640, 640]`), TFLite's GPU delegate will fail because it strictly requires NHWC (`[1, 640, 640, 3]`). EasyML's `InferenceDevice.AUTO` automatically falls back to **NNAPI**, which accelerates NCHW models at **30+ FPS** on hardware.

#### Q: Why are my bounding boxes not showing?
> **A:** Ensure your `confidenceThreshold` is not set too high (try `0.25f` to `0.40f`). Also verify that `CoordinateFormat.AUTO` is selected in your decoder so that normalized coordinate ranges `[0.0, 1.0]` are scaled correctly.

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
