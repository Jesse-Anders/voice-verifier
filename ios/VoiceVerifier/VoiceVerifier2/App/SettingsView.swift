import SwiftUI

struct SettingsView: View {
    @State private var serverURL: String = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL") ?? "http://127.0.0.1:8000"
    @State private var healthStatus: String = ""

    var body: some View {
        Form {
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
        }
        .navigationTitle("Settings")
    }
}


