// Tauri shell only — parked until the UI is actually needed (memory/action-log
// inspector, per CLAUDE.md §3.5). No commands exposed yet.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
