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

  // Utility function to smooth stroke points
  const smoothStrokePoints = useCallback((points: number[][]): number[][] => {
    if (points.length < 3) return points;

    // simple moving average smoothing (window size 3)
    const smoothed: number[][] = [];
    smoothed.push(points[0]);

    for (let i = 1; i < points.length - 1; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const next = points[i + 1];
      const smoothX = (prev[0] + curr[0] + next[0]) / 3;
      const smoothY = (prev[1] + curr[1] + next[1]) / 3;
      smoothed.push([smoothX, smoothY]);
    }

    smoothed.push(points[points.length - 1]);
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
        ctx.shadowColor = "rgba(0, 0, 0, 0.1)";
        ctx.shadowBlur = 1;
        ctx.shadowOffsetX = 0.5;
        ctx.shadowOffsetY = 0.5;

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
      const basePadding = Math.min(canvasWidth, canvasHeight) * 0.1;
      const lengthFactor = Math.min(textLength / 100, 1);
      const padding = basePadding + lengthFactor * basePadding * 0.5;

      const availableWidth = canvasWidth - 2 * padding;
      const availableHeight = canvasHeight - 2 * padding;

      const scaleX = availableWidth / actualWidth;
      const scaleY = availableHeight / actualHeight;
      const baseScale = Math.min(scaleX, scaleY);

      let scaleFactor = 0.85;

      // for shorter texts, use more space to make them larger and more readable
      if (textLength < 50) {
        scaleFactor = 0.95;
      } else if (textLength < 150) {
        scaleFactor = 0.9;
      } else if (textLength > 500) {
        // for very long texts, use slightly less space to ensure everything fits
        scaleFactor = 0.8;
      }

      const scale = baseScale * scaleFactor;

      // enhanced centering with vertical alignment adjustments
      const scaledWidth = actualWidth * scale;
      const scaledHeight = actualHeight * scale;
      const offsetX = (canvasWidth - scaledWidth) / 2 - minX * scale;

      // improved vertical centering slightly above center for better visual balance
      const verticalOffset = canvasHeight * 0.45; // 45% from top
      const offsetY = verticalOffset - (minY * scale + scaledHeight / 2);

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
