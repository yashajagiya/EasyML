package com.easyml.camera

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import com.easyml.detection.DetectionList

/**
 * Compose Canvas overlay that draws bounding boxes and labels on top of the camera preview.
 *
 * Automatically maps detection coordinates from model/image space to screen space.
 *
 * @param detections Current detection results (wrapped in [DetectionList] for compile-time stability)
 * @param imageWidth Width of the analyzed image
 * @param imageHeight Height of the analyzed image
 * @param modifier Compose modifier
 * @param boxColor Color for bounding boxes (default: green)
 * @param strokeWidth Width of box outlines
 * @param labelSize Font size for labels
 * @param isMirrored Set true for front camera to flip X coordinates
 */
@Composable
fun DetectionOverlay(
    detections: DetectionList,
    imageWidth: Int,
    imageHeight: Int,
    modifier: Modifier = Modifier,
    boxColor: Color = Color(0xFF00E676),
    strokeWidth: Float = 4f,
    labelSize: Int = 14,
    isMirrored: Boolean = false
) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(
        color = Color.White,
        fontSize = labelSize.sp
    )
    val bgColor = Color(0xAA000000)

    Canvas(modifier = modifier) {
        val scaleX = size.width / imageWidth.toFloat()
        val scaleY = size.height / imageHeight.toFloat()

        for (detection in detections) {
            val box = detection.boundingBox

            // Map coordinates to canvas space
            var left = box.left * scaleX
            var right = box.right * scaleX
            val top = box.top * scaleY
            val bottom = box.bottom * scaleY

            // Mirror for front camera
            if (isMirrored) {
                val tempLeft = size.width - right
                right = size.width - left
                left = tempLeft
            }

            val rectWidth = right - left
            val rectHeight = bottom - top

            // Draw bounding box
            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = Size(rectWidth, rectHeight),
                style = Stroke(width = strokeWidth)
            )

            // Draw label background + text
            val labelText = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val textLayoutResult = textMeasurer.measure(labelText, labelStyle)
            val textWidth = textLayoutResult.size.width.toFloat()
            val textHeight = textLayoutResult.size.height.toFloat()
            val padding = 4f

            // Background for label
            drawRect(
                color = bgColor,
                topLeft = Offset(left, top - textHeight - padding * 2),
                size = Size(textWidth + padding * 2, textHeight + padding * 2)
            )

            // Label text
            drawText(
                textMeasurer = textMeasurer,
                text = labelText,
                style = labelStyle,
                topLeft = Offset(left + padding, top - textHeight - padding)
            )
        }
    }
}
