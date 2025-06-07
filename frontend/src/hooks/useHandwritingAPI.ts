import { useState, useRef } from "react";
import { generateHandwriting } from "../services/api";

export const useHandwritingAPI = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [strokes, setStrokes] = useState<number[][]>([]);
  const errorTimeoutRef = useRef<number | null>(null);

  const setErrorWithTimeout = (errorMessage: string) => {
    // clear any existing timeout
    if (errorTimeoutRef.current) {
      clearTimeout(errorTimeoutRef.current);
    }

    setError(errorMessage);

    // clear error after 5 seconds
    errorTimeoutRef.current = setTimeout(() => {
      setError("");
    }, 5000);
  };

  const generate = async (text: string, bias: number = 2) => {
    if (!text.trim()) {
      setErrorWithTimeout("Please enter some text");
      return null;
    }

    setIsLoading(true);
    setError("");
    setStrokes([]);

    try {
      const response = await generateHandwriting(text, bias);

      if (response.success && response.strokes) {
        setStrokes(response.strokes);
        return response.strokes;
      } else {
        setErrorWithTimeout("No strokes received from server");
        return null;
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Network error occurred";
      setErrorWithTimeout(errorMessage);
      console.error("Generation error:", err);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const clearError = () => {
    setError("");
    if (errorTimeoutRef.current) {
      clearTimeout(errorTimeoutRef.current);
      errorTimeoutRef.current = null;
    }
  };

  const clearStrokes = () => setStrokes([]);

  return {
    generate,
    isLoading,
    error,
    strokes,
    clearError,
    clearStrokes,
  };
};
