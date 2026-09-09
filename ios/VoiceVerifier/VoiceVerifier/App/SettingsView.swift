import SwiftUI

// App settings screen
// - Lets the user set the server base URL used by all API clients
// - Provides a health check to ensure the server is reachable
// - Lets the user choose the segmentation mode (Amplitude vs Sliding)

struct SettingsView: View {
    @State private var serverURL: String = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL") ?? "http://127.0.0.1:8000"
    @State private var healthStatus: String = ""
    @State private var segMode: SegmentsAPI.SegmentationMode = SegmentsAPI.shared.currentMode()

    var body: some View {
        Form {
            // Server configuration (persisted in UserDefaults)
            Section(header: Text("Server")) {
                TextField("Server Base URL", text: $serverURL)
                    .keyboardType(.URL)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                Button("Save") {
                    UserDefaults.standard.set(serverURL, forKey: "EmbedAPIBaseURL")
                    EmbedAPI.shared.refreshBaseURL()
                }
                Button("Test Connection") {
                    let ok = EmbedAPI.shared.health()
                    healthStatus = ok ? "Connected" : "No response"
                }
                if !healthStatus.isEmpty { Text(healthStatus).font(.footnote).foregroundStyle(.secondary) }
            }
            // Segmentation mode: which server endpoint should be used to create segments
            Section(header: Text("Segmentation")) {
                Picker("Mode", selection: $segMode) {
                    ForEach(SegmentsAPI.SegmentationMode.allCases) { m in
                        Text(m.title).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: segMode) { newValue in
                    SegmentsAPI.shared.setMode(newValue)
                }
                Text({
                    switch segMode {
                    case .amplitude: return "Amplitude VAD (/segments)"
                    case .sliding: return "Sliding (VAD+Win/Hop) (/segments_sliding)"
                    }
                }())
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
        .onAppear { segMode = SegmentsAPI.shared.currentMode() }
    }
}


