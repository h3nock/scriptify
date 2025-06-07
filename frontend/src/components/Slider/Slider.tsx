import React from 'react'; 
import './Slider.css'; 

interface SliderProps {
    label: string; 
    value: number; 
    min: number; 
    max: number; 
    step: number; 
    onChange: (value: number) => void; 
    getLabel: (value: number) => string; 
    leftLabel: string; 
    rightLabel: string; 
    disabled?: boolean; 
    inverted?: boolean; 
}

const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  getLabel,
  leftLabel,
  rightLabel,
  disabled = false,
  inverted = false,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = Number(e.target.value);
    if (inverted) {
      // Invert the slider value by subtracting the new value from (min + max).
      // This reverses the slider's direction when the `inverted` prop is true.
      onChange((min + max) - newValue);
    } else {
      onChange(newValue);
    }
  };

  // Calculate the display value for the slider.
  // If `inverted` is true, reverse the value by subtracting it from (min + max).
  const displayValue = inverted ? (min + max) - value : value;

  return (
    <div className="setting-group">
      <label>
        {label}:{" "}
        <span className="setting-value">
          {getLabel(value)}
        </span>
      </label>
      <div className="slider-container">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={displayValue}
          onChange={handleChange}
          disabled={disabled}
          className="slider"
        />
        <div className="slider-labels">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      </div>
    </div>
  );
};

export default Slider; 