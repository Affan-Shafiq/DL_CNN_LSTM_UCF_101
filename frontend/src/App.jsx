import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictAction } from './api/client';
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
    } else {
      setError('Please select a valid video file');
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictAction(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to predict action. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <motion.div
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1>🎬 Action Recognition</h1>
        <p>Upload a video to identify the action being performed</p>
      </motion.div>

      <motion.div
        className="upload-card"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div
          className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => document.getElementById('file-input').click()}
        >
          <div className="upload-icon">📹</div>
          <h3>Drop your video here</h3>
          <p>or click to browse</p>
          <input
            id="file-input"
            type="file"
            accept="video/*"
            className="file-input"
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />
        </div>

        <AnimatePresence>
          {selectedFile && (
            <motion.div
              className="selected-file"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <div className="file-info">
                <span className="file-icon">🎥</span>
                <span className="file-name">{selectedFile.name}</span>
              </div>
              <button className="clear-btn" onClick={handleClear}>
                ✕
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {selectedFile && !isLoading && (
          <motion.button
            className="predict-btn"
            onClick={handlePredict}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Predict Action
          </motion.button>
        )}

        <AnimatePresence>
          {isLoading && (
            <motion.div
              className="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="spinner" />
              <p>Analyzing video...</p>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {result && (
            <motion.div
              className="result-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <div className="result-label">Detected Action</div>
              <div className="result-action">{result.action}</div>
              {result.confidence && (
                <div className="result-confidence">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {error && (
            <motion.div
              className="error-card"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

export default App;
