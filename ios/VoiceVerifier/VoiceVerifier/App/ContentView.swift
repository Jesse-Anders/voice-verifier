import SwiftUI

// Root/home view that lists enrolled speaker profiles and provides navigation.
// - Shows all profiles from ProfileStore
// - Taps navigate to RecordingView for that profile
// - Toolbar gear navigates to SettingsView
// - Provides an entry point to create a new profile via EnrollmentView

struct ContentView: View {
    @EnvironmentObject var store: ProfileStore

    var body: some View {
        NavigationStack {
            // Main list of actions
            List {
                Section("Profiles") {
                    ForEach(store.profiles) { p in
                        NavigationLink(destination: RecordingView(profile: p)) {
                            VStack(alignment: .leading) {
                                Text(p.name)
                                Text(String(format: "threshold %.2f", p.threshold)).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    // Swipe-to-delete removes the profile from storage
                    .onDelete(perform: store.remove)
                }

                Section {
                    NavigationLink("Enroll new profile", destination: EnrollmentView())
                }
            }
            .navigationTitle("Voice Verifier")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink(destination: SettingsView()) { Image(systemName: "gear") }
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View { ContentView().environmentObject(ProfileStore()) }
}


