import SwiftUI

struct ContentView: View {
    @EnvironmentObject var store: ProfileStore

    var body: some View {
        NavigationStack {
            List {
                Section("Profiles") {
                    ForEach(store.profiles) { p in
                        NavigationLink(destination: RecordingView(profile: p)) {
                            VStack(alignment: .leading) {
                                Text(p.name)
                                Text(String(format: "threshold %.2f", p.threshold)).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.onDelete(perform: store.remove)
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


