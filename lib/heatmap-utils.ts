/**
 * Utility functions for creating heatmap visualizations
 */

/**
 * Creates a heatmap overlay on a canvas context
 * 
 * @param ctx - The canvas 2D context
 * @param x - X coordinate (normalized 0-1)
 * @param y - Y coordinate (normalized 0-1)
 * @param width - Width (normalized 0-1)
 * @param height - Height (normalized 0-1)
 * @param canvasWidth - The canvas width in pixels
 * @param canvasHeight - The canvas height in pixels
 * @param color - Optional color string (default: "255, 0, 0" for red)
 * @param opacity - Optional opacity value (default: 0.7)
 */
export function createHeatmap(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  canvasWidth: number,
  canvasHeight: number,
  color: string = "255, 0, 0",
  opacity: number = 0.7
) {
  // Convert normalized coordinates to pixel values
  const xPos = x * canvasWidth;
  const yPos = y * canvasHeight;
  const boxWidth = width * canvasWidth;
  const boxHeight = height * canvasHeight;
  
  // Create gradient for heatmap effect
  const gradient = ctx.createRadialGradient(
    xPos + boxWidth / 2,
    yPos + boxHeight / 2,
    0,
    xPos + boxWidth / 2,
    yPos + boxHeight / 2,
    Math.max(boxWidth, boxHeight) / 1.5
  );

  // Add color stops for gradient
  gradient.addColorStop(0, `rgba(${color}, ${opacity})`);
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

  // Apply the gradient
  ctx.fillStyle = gradient;
  ctx.fillRect(xPos, yPos, boxWidth, boxHeight);
  
  // Draw bounding box
  ctx.strokeStyle = `rgba(${color}, 1)`;
  ctx.lineWidth = 2;
  ctx.strokeRect(xPos, yPos, boxWidth, boxHeight);
}

/**
 * Creates a simplified heatmap for PDF reports
 * This version works with direct image data without requiring a canvas
 */
export function createSimplifiedHeatmap(
  imageData: ImageData,
  x: number,
  y: number,
  width: number,
  height: number,
  color: string = "255, 0, 0",
  opacity: number = 0.7
): ImageData {
  // Create a copy of the image data
  const newData = new Uint8ClampedArray(imageData.data);
  const result = new ImageData(newData, imageData.width, imageData.height);
  
  // Convert normalized coordinates to pixel values
  const xStart = Math.floor(x * imageData.width);
  const yStart = Math.floor(y * imageData.height);
  const boxWidth = Math.floor(width * imageData.width);
  const boxHeight = Math.floor(height * imageData.height);
  
  // Parse color components
  const [r, g, b] = color.split(',').map(c => parseInt(c.trim(), 10));
  
  // Apply overlay to the region
  for (let py = yStart; py < yStart + boxHeight; py++) {
    for (let px = xStart; px < xStart + boxWidth; px++) {
      if (px < 0 || px >= imageData.width || py < 0 || py >= imageData.height) continue;
      
      // Calculate distance from center for gradient effect
      const centerX = xStart + boxWidth / 2;
      const centerY = yStart + boxHeight / 2;
      const maxDist = Math.max(boxWidth, boxHeight) / 2;
      const dist = Math.sqrt(Math.pow(px - centerX, 2) + Math.pow(py - centerY, 2));
      const factor = Math.max(0, 1 - dist / maxDist);
      
      // Calculate pixel index
      const idx = (py * imageData.width + px) * 4;
      
      // Blend colors based on distance and opacity
      const blendFactor = factor * opacity;
      result.data[idx] = Math.min(255, (1 - blendFactor) * result.data[idx] + blendFactor * r);
      result.data[idx + 1] = Math.min(255, (1 - blendFactor) * result.data[idx + 1] + blendFactor * g);
      result.data[idx + 2] = Math.min(255, (1 - blendFactor) * result.data[idx + 2] + blendFactor * b);
      // Alpha channel remains unchanged
    }
  }
  
  return result;
} 