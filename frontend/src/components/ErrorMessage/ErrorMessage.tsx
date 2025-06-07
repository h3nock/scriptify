import './ErrorMessage.css';

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage = ({ message }: ErrorMessageProps) => {
  if (!message) return null;

  return (
    <div className="error-message">
      <span className="error-icon">⚠️</span>
      {message}
    </div>
  );
};

export default ErrorMessage;