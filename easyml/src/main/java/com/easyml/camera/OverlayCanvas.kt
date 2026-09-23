package com.easyml.camera

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
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
 * @param boxColor Color for bounding boxes (default: vibrant green)
 * @param strokeWidth Width of box outlines (default: 4f)
 * @param cornerRadius Rounded corner radius for bounding boxes and label tags (default: 8f)
 * @param labelSize Font size for labels (default: 14sp)
 * @param showLabels Whether to show class name on the bounding box (default: true)
 * @param showConfidence Whether to show confidence percentage (default: true)
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
    cornerRadius: Float = 8f,
    labelSize: Int = 14,
    showLabels: Boolean = true,
    showConfidence: Boolean = true,
    isMirrored: Boolean = false
) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(
        color = Color.White,
        fontSize = labelSize.sp
    )
    val bgColor = Color(0xAA000000)

    Canvas(modifier = modifier) {
        if (imageWidth <= 0 || imageHeight <= 0) return@Canvas

        // Map from analyzed image coordinates to PreviewView FILL_CENTER space
        val scale = maxOf(size.width / imageWidth.toFloat(), size.height / imageHeight.toFloat())
        val offsetX = (size.width - imageWidth * scale) / 2f
        val offsetY = (size.height - imageHeight * scale) / 2f

        for (detection in detections) {
            val box = detection.boundingBox

            // Map coordinates to canvas space
            var left = box.left * scale + offsetX
            var right = box.right * scale + offsetX
            val top = box.top * scale + offsetY
            val bottom = box.bottom * scale + offsetY

            // Mirror for front camera
            if (isMirrored) {
                val tempLeft = size.width - right
                right = size.width - left
                left = tempLeft
            }

            val rectWidth = right - left
            val rectHeight = bottom - top

            // Draw bounding box
            if (cornerRadius > 0f) {
                drawRoundRect(
                    color = boxColor,
                    topLeft = Offset(left, top),
                    size = Size(rectWidth, rectHeight),
                    cornerRadius = CornerRadius(cornerRadius, cornerRadius),
                    style = Stroke(width = strokeWidth)
                )
            } else {
                drawRect(
                    color = boxColor,
                    topLeft = Offset(left, top),
                    size = Size(rectWidth, rectHeight),
                    style = Stroke(width = strokeWidth)
                )
            }

            // Draw label background + text if requested
            if (showLabels || showConfidence) {
                val labelText = when {
                    showLabels && showConfidence -> "${detection.label} ${(detection.confidence * 100).toInt()}%"
                    showLabels -> detection.label
                    else -> "${(detection.confidence * 100).toInt()}%"
                }

                val textLayoutResult = textMeasurer.measure(labelText, labelStyle)
                val textWidth = textLayoutResult.size.width.toFloat()
                val textHeight = textLayoutResult.size.height.toFloat()
                val padding = 6f
                val labelTop = maxOf(0f, top - textHeight - padding * 2)

                // Background for label
                if (cornerRadius > 0f) {
                    drawRoundRect(
                        color = bgColor,
                        topLeft = Offset(left, labelTop),
                        size = Size(textWidth + padding * 2, textHeight + padding * 2),
                        cornerRadius = CornerRadius(cornerRadius / 2, cornerRadius / 2)
                    )
                } else {
                    drawRect(
                        color = bgColor,
                        topLeft = Offset(left, labelTop),
                        size = Size(textWidth + padding * 2, textHeight + padding * 2)
                    )
                }

                // Label text
                drawText(
                    textMeasurer = textMeasurer,
                    text = labelText,
                    style = labelStyle,
                    topLeft = Offset(left + padding, labelTop + padding)
                )
            }
        }
    }
}
