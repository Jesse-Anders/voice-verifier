import SwiftUI

// Application entry point
// - Creates a shared ProfileStore and injects it into the environment
// - Configures startup behavior (e.g., skip loading local Torch model)

@main
struct VoiceVerifierApp: App {
    @StateObject private var store = ProfileStore()

    init() {
        // Use server embeddings only: skip local Torch model load to speed startup
        UserDefaults.standard.set(true, forKey: "SkipLocalModel")
        // To revert later, comment the line above and optionally run:
        // UserDefaults.standard.removeObject(forKey: "SkipLocalModel")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
    }
}


