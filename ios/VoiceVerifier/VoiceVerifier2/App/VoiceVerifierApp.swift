import SwiftUI

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


