import { useState } from "react";
import "./TextInput.css";

interface TextInputProps {
  onTextChange: (text: string) => void;
  onEnterPress?: () => void;
  maxLength?: number;
  placeholder?: string;
  hasError?: boolean;
  disabled?: boolean;
}

const TextInput = ({
  onTextChange,
  onEnterPress,
  maxLength = 30,
  placeholder = "Type your message to convert to handwriting...",
  hasError = false,
  disabled = false,
}: TextInputProps) => {
  const [text, setText] = useState("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newText = event.target.value;

    if (newText.length <= maxLength) {
      setText(newText);
      onTextChange(newText);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !event.shiftKey && onEnterPress) {
      event.preventDefault();
      onEnterPress();
    }
  };

  return (
    <div className="text-input-container">
      <div className="input-wrapper">
        <input
          type="text"
          value={text}
          onChange={handleInputChange}
          onKeyDown={handleKeyPress}
          placeholder={placeholder}
          className={`text-input ${hasError ? "error" : ""} ${
            disabled ? "disabled" : ""
          }`}
          disabled={disabled}
          maxLength={maxLength}
        />
        <div className="character-count">
          {text.length}/{maxLength}
        </div>
      </div>
      <div className="helper-text">
        Note: Input is limited to {maxLength} characters to ensure optimal performance on free-tier hosting.
      </div>
    </div>
  );
};

export default TextInput;
