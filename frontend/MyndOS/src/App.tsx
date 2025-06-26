import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { motion, AnimatePresence } from "framer-motion";

interface TranscriptionResult {
  success: boolean;
  text: string;
  error?: string;
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
  const ALLOWED_TYPES = ["audio/mp3", "audio/wav", "audio/mpeg", "audio/wave"];

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Please select a valid audio file (.mp3 or .wav)";
    }
    
    if (file.size > MAX_FILE_SIZE) {
      return "File size must be less than 100MB";
    }
    
    return null;
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }

    setError("");
    setSelectedFile(file);
    setTranscription("");
  };

  const handleTranscribe = async () => {
    if (!selectedFile) {
      setError("Please select an audio file first");
      return;
    }

    setIsLoading(true);
    setError("");
    setTranscription("");

    try {
      // Convert file to bytes
      const fileBytes = await selectedFile.arrayBuffer();
      const fileData = new Uint8Array(fileBytes);

      // Save file to backend
      const savedFilePath = await invoke<string>("save_audio_file", {
        fileData: Array.from(fileData),
        filename: selectedFile.name,
      });

      // Transcribe the audio
      const result = await invoke<TranscriptionResult>("transcribe_audio", {
        filePath: savedFilePath,
      });

      if (result.success) {
        setTranscription(result.text);
      } else {
        setError(result.error || "Transcription failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      setSelectedFile(file);
      setError("");
      setTranscription("");
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const clearAll = () => {
    setSelectedFile(null);
    setTranscription("");
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-light to-secondary-light dark:from-dark-light dark:to-dark-dark flex flex-col">
      {/* Navbar */}
      <nav className="glass-card mx-4 mt-4 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-light to-secondary-light">MyndOS</h1>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm mr-2">Status:</span>
          <span className={`h-3 w-3 rounded-full ${isLoading ? 'bg-yellow-400 animate-pulse' : 'bg-green-500'}`}></span>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center p-4 md:p-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card w-full max-w-3xl overflow-hidden"
        >
          {/* Upload Section */}
          <div className="p-6">
            <motion.div 
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${selectedFile ? 'border-green-500 bg-green-50/20' : 'border-gray-300 hover:border-primary-light hover:bg-primary-light/5'}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              <AnimatePresence mode="wait">
                {selectedFile ? (
                  <motion.div 
                    key="file-info"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center justify-center space-x-4"
                  >
                    <div className="text-4xl">🎵</div>
                    <div className="text-left">
                      <div className="font-semibold text-lg">{selectedFile.name}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div 
                    key="upload-prompt"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center space-y-3"
                  >
                    <div className="text-5xl mb-2">📁</div>
                    <p className="text-lg">Click to select or drag and drop an audio file</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Supports .mp3 and .wav files (max 100MB)</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Error Message */}
            <AnimatePresence>
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-4 bg-red-100 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center space-x-2"
                >
                  <span className="text-xl">⚠️</span>
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Transcription Result */}
            <AnimatePresence>
              {transcription && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-6 bg-white/50 dark:bg-dark-light/50 rounded-lg p-4"
                >
                  <h3 className="text-lg font-semibold mb-2">Transcription Result</h3>
                  <div className="bg-white dark:bg-dark-dark p-4 rounded-md border border-gray-200 dark:border-gray-700 max-h-60 overflow-y-auto whitespace-pre-wrap">
                    {transcription}
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="btn-success mt-3 text-sm py-2 px-4"
                    onClick={() => navigator.clipboard.writeText(transcription)}
                  >
                    📋 Copy to Clipboard
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </main>

      {/* Bottom Bar */}
      <footer className="glass-card mx-4 mb-4 p-4 flex justify-center items-center space-x-4">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="btn-primary flex items-center space-x-2"
          onClick={handleTranscribe}
          disabled={!selectedFile || isLoading}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Transcribing...</span>
            </>
          ) : (
            <span>Transcribe</span>
          )}
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="btn-secondary"
          onClick={clearAll}
          disabled={isLoading || (!selectedFile && !transcription && !error)}
        >
          Clear
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="btn-secondary"
          onClick={() => setShowLogs(!showLogs)}
        >
          Logs
        </motion.button>
      </footer>

      {/* Logs Panel */}
      <AnimatePresence>
        {showLogs && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-20 left-4 right-4 glass-card p-4 max-h-60 overflow-y-auto"
          >
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">Application Logs</h3>
              <button 
                onClick={() => setShowLogs(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                ✕
              </button>
            </div>
            <div className="text-xs font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
              {/* Sample logs - in a real app, these would be actual logs */}
              <p className="text-gray-500">[System] Application started</p>
              {selectedFile && (
                <p className="text-blue-500">[Info] File selected: {selectedFile.name}</p>
              )}
              {isLoading && (
                <p className="text-yellow-500">[Process] Transcription in progress...</p>
              )}
              {transcription && (
                <p className="text-green-500">[Success] Transcription completed</p>
              )}
              {error && (
                <p className="text-red-500">[Error] {error}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
