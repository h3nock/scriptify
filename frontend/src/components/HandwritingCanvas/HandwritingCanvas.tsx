import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";
import "./HandwritingCanvas.css";

interface HandwritingCanvasProps {
  strokes: number[][];
  animationSpeed?: number;
  strokeColor?: string;
  strokeWidth?: number;
  canvasWidth?: number;
  canvasHeight?: number;
  showControls?: boolean;
}

const HandwritingCanvas: React.FC<HandwritingCanvasProps> = ({
  strokes,
  animationSpeed = 20,
  strokeColor = "#2c3e50",
  strokeWidth = 2.8,
  canvasWidth = 800,
  canvasHeight = 200,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // animation state
  const [currentStrokeIndex, setCurrentStrokeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Chaikin's smoothing algorithm
  const smoothStrokePoints = useCallback((points: number[][]): number[][] => {
    if (points.length < 3) return points;

    let smoothed = [...points];
    // Apply 2 iterations of Chaikin's algorithm for smoothness
    for (let iter = 0; iter < 2; iter++) {
      const newPoints: number[][] = [];
      newPoints.push(smoothed[0]); // Keep start point

      for (let i = 0; i < smoothed.length - 1; i++) {
        const p0 = smoothed[i];
        const p1 = smoothed[i + 1];

        // Q = 0.75*P0 + 0.25*P1
        // R = 0.25*P0 + 0.75*P1
        const qx = 0.75 * p0[0] + 0.25 * p1[0];
        const qy = 0.75 * p0[1] + 0.25 * p1[1];

        const rx = 0.25 * p0[0] + 0.75 * p1[0];
        const ry = 0.25 * p0[1] + 0.75 * p1[1];

        newPoints.push([qx, qy]);
        newPoints.push([rx, ry]);
      }

      newPoints.push(smoothed[smoothed.length - 1]); // Keep end point
      smoothed = newPoints;
    }

    return smoothed;
  }, []);

  // helper function to draw smooth strokes using quadratic curves
  const drawStrokes = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      strokesToDraw: number[][],
      upToIndex?: number
    ) => {
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      if (strokesToDraw.length === 0) return;

      ctx.strokeStyle = strokeColor;
      // Dynamic stroke width based on canvas size to ensure visibility
      // but respecting the user's choice
      ctx.lineWidth = strokeWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalCompositeOperation = "source-over";

      // enable anti-aliasing for smoother lines
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";

      let currentPathPoints: number[][] = [];
      const pointsToDrawCount =
        upToIndex !== undefined ? upToIndex : strokesToDraw.length;

      // helper function to draw smooth curves through points
      const drawSmoothPath = (points: number[][]) => {
        if (points.length < 2) return;

        // apply smoothing to the points
        const smoothedPoints = smoothStrokePoints(points);

        // add subtle shadow effect for depth
        ctx.save();
        ctx.shadowColor = "rgba(0, 0, 0, 0.15)"; // Slightly darker shadow
        ctx.shadowBlur = 2; // Increased blur
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;

        ctx.beginPath();
        ctx.moveTo(smoothedPoints[0][0], smoothedPoints[0][1]);

        if (smoothedPoints.length === 2) {
          // for just two points, draw a straight line
          ctx.lineTo(smoothedPoints[1][0], smoothedPoints[1][1]);
        } else {
          // for multiple points, use quadratic curves for smoothness
          for (let i = 1; i < smoothedPoints.length - 1; i++) {
            const currentPoint = smoothedPoints[i];
            const nextPoint = smoothedPoints[i + 1];

            // calculate control point for smooth curve
            const controlX = (currentPoint[0] + nextPoint[0]) / 2;
            const controlY = (currentPoint[1] + nextPoint[1]) / 2;

            ctx.quadraticCurveTo(
              currentPoint[0],
              currentPoint[1],
              controlX,
              controlY
            );
          }

          // draw to the final point
          const lastPoint = smoothedPoints[smoothedPoints.length - 1];
          ctx.lineTo(lastPoint[0], lastPoint[1]);
        }

        ctx.stroke();
        ctx.restore();
      };

      for (
        let i = 0;
        i < Math.min(pointsToDrawCount, strokesToDraw.length);
        i++
      ) {
        const strokePoint = strokesToDraw[i];

        if (!Array.isArray(strokePoint) || strokePoint.length < 3) {
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
          continue;
        }

        const [x, y, penState] = strokePoint;

        if (!isFinite(x) || !isFinite(y)) {
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
          continue;
        }

        currentPathPoints.push([x, y]);

        if (penState === 1) {
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
        }
      }

      // draw any remaining path
      if (currentPathPoints.length > 1) {
        drawSmoothPath(currentPathPoints);
      }
    },
    [canvasWidth, canvasHeight, strokeColor, strokeWidth, smoothStrokePoints]
  );
  const normalizedStrokes = useMemo(() => {
    if (strokes.length === 0) return [];

    try {
      // find bounds from all coordinates
      let minX = Infinity,
        maxX = -Infinity;
      let minY = Infinity,
        maxY = -Infinity;
      let validPoints = 0;

      strokes.forEach((stroke) => {
        if (!Array.isArray(stroke) || stroke.length < 2) return;
        const [x, y] = stroke;
        if (
          typeof x === "number" &&
          typeof y === "number" &&
          isFinite(x) &&
          isFinite(y)
        ) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
          validPoints++;
        }
      });

      if (validPoints === 0 || !isFinite(minX)) {
        console.warn("No valid stroke points found");
        return [];
      }

      const strokeWidthDim = maxX - minX;
      const strokeHeightDim = maxY - minY;

      const minDimension = 10;
      const actualWidth = Math.max(strokeWidthDim, minDimension);
      const actualHeight = Math.max(strokeHeightDim, minDimension);

      // dynamic padding based on text length and canvas size
      const textLength = strokes.length;
      const basePadding = Math.min(canvasWidth, canvasHeight) * 0.05; // Reduced base padding

      // Continuous scaling factor based on text length
      // For short text, we want it larger (closer to 0.95)
      // For long text, we want it smaller but not too small (down to 0.75)
      // Using a decay function: scale = minScale + (maxScale - minScale) * e^(-k * length)
      const maxScale = 0.95;
      const minScale = 0.75;
      const decayRate = 0.002; // Adjust this to control how fast it shrinks
      
      let scaleFactor = minScale + (maxScale - minScale) * Math.exp(-decayRate * textLength);
      
      // Clamp scale factor just in case
      scaleFactor = Math.max(minScale, Math.min(maxScale, scaleFactor));

      const availableWidth = canvasWidth - 2 * basePadding;
      // Reserve extra space at the bottom for the controls (approx 60px)
      const bottomPadding = basePadding + 60;
      const availableHeight = canvasHeight - basePadding - bottomPadding;

      const scaleX = availableWidth / actualWidth;
      const scaleY = availableHeight / actualHeight;
      
      // Use the smaller scale to fit both dimensions, then apply our continuous factor
      const baseScale = Math.min(scaleX, scaleY);
      const scale = baseScale * scaleFactor;

      // enhanced centering with vertical alignment adjustments
      const scaledWidth = actualWidth * scale;
      const scaledHeight = actualHeight * scale;
      const offsetX = (canvasWidth - scaledWidth) / 2 - minX * scale;

      // improved vertical centering
      // We center within the available height (top to bottom-padding)
      // effectively pushing content up
      // Shift up by an additional 40px to ensure it clears the bottom controls comfortably
      const verticalCenter = basePadding + availableHeight / 2 + 40;
      const offsetY = verticalCenter - (minY * scale + scaledHeight / 2);

      const normalized = strokes.map((stroke, index) => {
        if (!Array.isArray(stroke) || stroke.length < 2) {
          console.warn(`Invalid stroke at index ${index}:`, stroke);
          return [canvasWidth / 2, canvasHeight / 2, 1];
        }
        const [x, y, penState = 0] = stroke;

        const normalizedX = (typeof x === "number" ? x : 0) * scale + offsetX;
        const normalizedY =
          canvasHeight - ((typeof y === "number" ? y : 0) * scale + offsetY);

        return [normalizedX, normalizedY, penState];
      });

      return normalized;
    } catch (error) {
      console.error("Error normalizing strokes:", error);
      return [];
    }
  }, [strokes, canvasWidth, canvasHeight]);

  // auto start animation when new strokes are received
  useEffect(() => {
    if (normalizedStrokes.length > 0) {
      setCurrentStrokeIndex(0);
      setIsPlaying(true);
    } else {
      // no strokes, clear canvas and reset state
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      setCurrentStrokeIndex(0);
      setIsPlaying(false);
    }
  }, [normalizedStrokes]);

  // animation loop with consistent timing
  useEffect(() => {
    if (!isPlaying || currentStrokeIndex >= normalizedStrokes.length) {
      if (currentStrokeIndex >= normalizedStrokes.length && isPlaying) {
        setIsPlaying(false); // stop animation when complete
      }
      return;
    }

    const timer = setTimeout(() => {
      setCurrentStrokeIndex((prev) => prev + 1);
    }, animationSpeed);

    return () => clearTimeout(timer);
  }, [isPlaying, currentStrokeIndex, normalizedStrokes.length, animationSpeed]);

  // re-animate function
  const reAnimate = useCallback(() => {
    if (normalizedStrokes.length > 0) {
      setCurrentStrokeIndex(0);
      setIsPlaying(true);
    }
  }, [normalizedStrokes]);

  const downloadCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // create a temporary canvas with higher resolution for better quality
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return;

    // Set high resolution
    const scale = 3;
    tempCanvas.width = canvasWidth * scale;
    tempCanvas.height = canvasHeight * scale;

    // scale the context to ensure correct drawing operations
    tempCtx.scale(scale, scale);

    // set white background
    tempCtx.fillStyle = "#ffffff";
    tempCtx.fillRect(0, 0, canvasWidth, canvasHeight);

    tempCtx.strokeStyle = strokeColor;
    // adaptive stroke width for download
    const downloadStrokeWidth = strokeWidth * 1.2;
    tempCtx.lineWidth = downloadStrokeWidth;
    tempCtx.lineCap = "round";
    tempCtx.lineJoin = "round";
    tempCtx.globalCompositeOperation = "source-over";
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = "high";

    // helper function for smooth curves
    const drawSmoothPath = (points: number[][]) => {
      if (points.length < 2) return;

      tempCtx.beginPath();
      tempCtx.moveTo(points[0][0], points[0][1]);

      if (points.length === 2) {
        tempCtx.lineTo(points[1][0], points[1][1]);
      } else {
        for (let i = 1; i < points.length - 1; i++) {
          const currentPoint = points[i];
          const nextPoint = points[i + 1];

          const controlX = (currentPoint[0] + nextPoint[0]) / 2;
          const controlY = (currentPoint[1] + nextPoint[1]) / 2;

          tempCtx.quadraticCurveTo(
            currentPoint[0],
            currentPoint[1],
            controlX,
            controlY
          );
        }

        const lastPoint = points[points.length - 1];
        tempCtx.lineTo(lastPoint[0], lastPoint[1]);
      }

      tempCtx.stroke();
    };

    // draw all strokes with smooth curves
    if (normalizedStrokes.length > 0) {
      let currentPathPoints: number[][] = [];

      for (let i = 0; i < normalizedStrokes.length; i++) {
        const strokePoint = normalizedStrokes[i];

        if (!Array.isArray(strokePoint) || strokePoint.length < 2) {
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
          continue;
        }

        const [x, y, penState = 0] = strokePoint;

        if (!isFinite(x) || !isFinite(y)) {
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
          continue;
        }

        currentPathPoints.push([x, y]);
        if (penState === 1) {
          // pen up -> draw current path if it has multiple points, then reset
          if (currentPathPoints.length > 1) {
            drawSmoothPath(currentPathPoints);
          }
          currentPathPoints = [];
        }
      }
    }

    // download the image
    tempCanvas.toBlob(
      (blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `handwriting-${Date.now()}.png`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        }
      },
      "image/png",
      1.0
    );
  }, [normalizedStrokes, canvasWidth, canvasHeight, strokeColor, strokeWidth]);

  // event listeners for replay and download
  useEffect(() => {
    const handleReplay = () => reAnimate();
    const handleDownload = () => downloadCanvas();

    document.addEventListener("replayCanvas", handleReplay);
    document.addEventListener("downloadCanvas", handleDownload);

    return () => {
      document.removeEventListener("replayCanvas", handleReplay);
      document.removeEventListener("downloadCanvas", handleDownload);
    };
  }, [reAnimate, downloadCanvas]);

  // canvas setup
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set up high DPI rendering
    const devicePixelRatio = window.devicePixelRatio || 1;
    const displayWidth = canvasWidth;
    const displayHeight = canvasHeight;

    // set actual canvas size based on device pixel ratio
    canvas.width = displayWidth * devicePixelRatio;
    canvas.height = displayHeight * devicePixelRatio;

    // scale canvas back down using CSS
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";

    // scale the drawing context to match device pixel ratio
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const pointsToDrawCount = isPlaying
      ? currentStrokeIndex
      : normalizedStrokes.length;
    drawStrokes(ctx, normalizedStrokes, pointsToDrawCount);
  }, [
    currentStrokeIndex,
    normalizedStrokes,
    isPlaying,
    drawStrokes,
    canvasWidth,
    canvasHeight,
  ]);

  // handle empty strokes
  if (strokes.length === 0 && !isPlaying) {
    return (
      <div className="handwriting-canvas-container">
        <div className="canvas-placeholder">
          <p>Generated handwriting will appear here</p>
        </div>
      </div>
    );
  }
  return (
    <div className="handwriting-canvas-container">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        className="handwriting-canvas"
      />

      {normalizedStrokes.length > 0 && (
        <div className="canvas-controls">
          <button
            onClick={reAnimate}
            disabled={!normalizedStrokes.length || isPlaying}
            className="control-btn animate-btn"
          >
            {isPlaying ? "Animating..." : "Replay"}
          </button>

          <button
            onClick={downloadCanvas}
            disabled={!normalizedStrokes.length}
            className="control-btn download-btn"
            title="Download handwriting as PNG image"
          >
            Download
          </button>
        </div>
      )}
    </div>
  );
};

export default HandwritingCanvas;
