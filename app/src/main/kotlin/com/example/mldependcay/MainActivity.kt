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
import com.example.mldependcay.ui.theme.MlDependcayTheme

/**
 * Demo app showing how simple EasyML makes it to run YOLO object detection.
 *
 * The ENTIRE detection setup is just 5 lines:
 * 1. Create detector with model + labels
 * 2. Put EasyMLCameraView in your Compose layout
 * That's it!
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MlDependcayTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    DetectionScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun DetectionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current

    // ✅ That's it! Just 5 lines to set up YOLO detection:
    val detector = remember {
        EasyML.objectDetector(context) {
            model = ModelSource.Asset("yolon.tflite")
            labels = LabelSource.Asset("labels.txt")
            confidenceThreshold = 0.4f
            device = InferenceDevice.AUTO  // Automatically selects GPU FP16 or XNNPACK CPU for best speed
        }
    }

    // Clean up when composable leaves composition
    DisposableEffect(Unit) {
        onDispose { detector.close() }
    }

    // ✅ One composable for live camera + detection + bounding boxes + FPS
    EasyMLCameraView(
        detector = detector,
        modifier = modifier,
        showOverlay = true,
        showFps = true,
        onResults = { detections ->
            // Optional: do something custom with results
            // e.g., log them, update a ViewModel, trigger an action
        }
    )
}