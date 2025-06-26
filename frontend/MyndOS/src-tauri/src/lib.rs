// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::fs;
use std::path::Path;
use std::process::Command;
use std::env;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TranscriptionResult {
    success: bool,
    text: String,
    error: Option<String>,
}

#[tauri::command]
async fn save_audio_file(file_data: Vec<u8>, filename: String) -> Result<String, String> {
    // Use a simple temp directory approach
    let temp_dir = env::temp_dir().join("myndos_audio");
    fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp directory: {}", e))?;
    
    // Save the file
    let file_path = temp_dir.join(&filename);
    fs::write(&file_path, file_data)
        .map_err(|e| format!("Failed to save file: {}", e))?;
    
    Ok(file_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn transcribe_audio(file_path: String) -> Result<TranscriptionResult, String> {
    // Validate file exists
    if !Path::new(&file_path).exists() {
        return Err("Audio file not found".to_string());
    }
    
    // Get the Python script path (assuming it's in the backend directory)
    let current_exe = std::env::current_exe()
        .map_err(|e| format!("Failed to get current executable path: {}", e))?;
    
    let app_dir = current_exe.parent()
        .ok_or("Failed to get app directory")?;
    
    // Navigate to the backend directory relative to the app
    let backend_dir = app_dir.join("backend");
    let python_script = backend_dir.join("transcribe_audio.py");
    
    if !python_script.exists() {
        return Err("Python transcription script not found".to_string());
    }
    
    // Run the Python script
    let output = Command::new("python")
        .arg(python_script.to_string_lossy().to_string())
        .arg(&file_path)
        .current_dir(&backend_dir)
        .output()
        .map_err(|e| format!("Failed to execute Python script: {}", e))?;
    
    if output.status.success() {
        let transcription = String::from_utf8(output.stdout)
            .map_err(|e| format!("Failed to parse transcription output: {}", e))?
            .trim()
            .to_string();
        
        Ok(TranscriptionResult {
            success: true,
            text: transcription,
            error: None,
        })
    } else {
        let error_output = String::from_utf8(output.stderr)
            .unwrap_or_else(|_| "Unknown error".to_string());
        
        Ok(TranscriptionResult {
            success: false,
            text: String::new(),
            error: Some(error_output),
        })
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            save_audio_file,
            transcribe_audio
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
