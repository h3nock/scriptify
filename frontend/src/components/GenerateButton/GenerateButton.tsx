import "./GenerateButton.css";

interface GenerateButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isLoading?: boolean;
}

const GenerateButton = ({
  onClick,
  disabled = false,
  isLoading = false,
}: GenerateButtonProps) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className="generate-button"
      aria-label={isLoading ? "Generating handwriting" : "Generate handwriting"}
    >
      {isLoading ? (
        <>
          <span className="loading-spinner"></span>
          Generating...
        </>
      ) : (
        "Generate"
      )}
    </button>
  );
};

export default GenerateButton;
