import { useState, useEffect } from "react";
import TextInput from "./components/TextInput/TextInput";
import GenerateButton from "./components/GenerateButton/GenerateButton";
import ErrorMessage from "./components/ErrorMessage/ErrorMessage";
import HandwritingCanvas from "./components/HandwritingCanvas/HandwritingCanvas";
import Slider from "./components/Slider/Slider";
import { useHandwritingAPI } from "./hooks/useHandwritingAPI";
import "./App.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [bias, setBias] = useState(2); // default style value
  const [animationSpeed, setAnimationSpeed] = useState(20); // default speed: 20ms
  const [strokeWidth, setStrokeWidth] = useState(4); // default stroke width: 4
  const [showSettings, setShowSettings] = useState(false);

  // temporary state for settings popup
  const [tempBias, setTempBias] = useState(bias);
  const [tempAnimationSpeed, setTempAnimationSpeed] = useState(animationSpeed);
  const [tempStrokeWidth, setTempStrokeWidth] = useState(strokeWidth);

  const { generate, isLoading, error, strokes, clearError } =
    useHandwritingAPI();

  // keyboard shortcuts for settings popup
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && showSettings) {
        closeSettings();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [showSettings]);

  const handleTextChange = (text: string) => {
    setInputText(text);
    if (error) clearError(); // clear error when user starts typing
  };

  // Style label function 
  const getStyleLabel = (value: number) => {
    if (value <= 1) return "Natural";
    if (value < 2) return "Balanced";
    return "Clean";
  };

  // Speed label function 
  const getSpeedLabel = (value: number) => {
    if (value >= 25) return "Slow";
    if (value >= 15) return "Normal";
    return "Fast";
  };

  // Stroke width label function
  const getStrokeWidthLabel = (value: number) => {
    if (value <= 2.5) return "Thin";
    if (value <= 4) return "Normal";
    if (value <= 7) return "Medium";
    return "Thick";
  };

  const openSettings = () => {
    setTempBias(bias);
    setTempAnimationSpeed(animationSpeed);
    setTempStrokeWidth(strokeWidth);
    setShowSettings(true);
  };

  const closeSettings = () => {
    setShowSettings(false);
  };

  const saveSettings = () => {
    setBias(tempBias);
    setAnimationSpeed(tempAnimationSpeed);
    setStrokeWidth(tempStrokeWidth);
    setShowSettings(false);
  };

  const handleGenerate = async () => {
    if (inputText.trim().length === 0) return;

    await generate(inputText, bias);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>Scriptify</h1>
          <div className="header-controls">
            <button
              className={`settings-toggle ${showSettings ? "active" : ""}`}
              onClick={openSettings}
              aria-label="Toggle settings"
            >
              ⚙
            </button>
          </div>
        </div>
      </header>

      {/* Settings Popup */}
      {showSettings && (
        <>
          <div className="settings-backdrop" onClick={closeSettings} />
          <div className="settings-popup">
            <div className="settings-popup-header">
              <h3>Settings</h3>
              <button
                className="close-button"
                onClick={closeSettings}
                aria-label="Close settings"
              >
                ×
              </button>
            </div>
            <div className="settings-popup-content">
              <Slider
                label="Style"
                value={tempBias}
                min={0.5}
                max={6}
                step={0.1}
                onChange={setTempBias}
                getLabel={getStyleLabel}
                leftLabel="Natural"
                rightLabel="Clean"
                disabled={isLoading}
              />

              <Slider
                label="Speed"
                value={tempAnimationSpeed}
                min={1}
                max={40}
                step={1}
                onChange={setTempAnimationSpeed}
                getLabel={getSpeedLabel}
                leftLabel="Slow"
                rightLabel="Fast"
                disabled={isLoading}
                inverted={true}
              />

              <Slider
                label="Stroke Width"
                value={tempStrokeWidth}
                min={1}
                max={10}
                step={0.1}
                onChange={setTempStrokeWidth}
                getLabel={getStrokeWidthLabel}
                leftLabel="Thin"
                rightLabel="Thick"
                disabled={isLoading}
              />

              <div className="settings-popup-actions">
                <button className="cancel-button" onClick={closeSettings}>
                  Cancel
                </button>
                <button className="save-button" onClick={saveSettings}>
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      <main className="chat-container">
        {/* Canvas Area */}
        <div className="canvas-area">
          <HandwritingCanvas
            strokes={strokes}
            animationSpeed={animationSpeed}
            strokeColor="#2c3e50"
            strokeWidth={strokeWidth}
            canvasWidth={800}
            canvasHeight={300}
            showControls={strokes.length > 0}
          />
        </div>

        {/* Input Area */}
        <div className="input-area">
          {error && <ErrorMessage message={error} />}

          <div className="input-container">
            <TextInput
              onTextChange={handleTextChange}
              onEnterPress={handleGenerate}
              hasError={!!error}
              disabled={isLoading}
              maxLength={30}
              placeholder="Type your message..."
            />
            <GenerateButton
              onClick={handleGenerate}
              disabled={inputText.trim().length === 0 || isLoading}
              isLoading={isLoading}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
