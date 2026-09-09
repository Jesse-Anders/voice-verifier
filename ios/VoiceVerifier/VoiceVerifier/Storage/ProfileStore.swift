import Foundation

final class ProfileStore: ObservableObject {
    @Published private(set) var profiles: [SpeakerVerifier.Profile] = []

    private let fileURL: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.fileURL = docs.appendingPathComponent("profiles.json")
        self.profiles = (try? Self.load(from: fileURL)) ?? []
    }

    func add(name: String, centroid: [Float], threshold: Float = 0.65, trainingFiles: [String]? = nil) {
        let profile = SpeakerVerifier.Profile(id: UUID(), name: name, centroid: centroid, threshold: threshold, trainingFiles: trainingFiles)
        profiles.append(profile)
        save()
    }

    func update(profile: SpeakerVerifier.Profile) {
        if let idx = profiles.firstIndex(where: { $0.id == profile.id }) {
            profiles[idx] = profile
            save()
        }
    }

    func remove(at offsets: IndexSet) {
        profiles.remove(atOffsets: offsets)
        save()
    }

    private func save() {
        do {
            let data = try JSONEncoder().encode(profiles)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("profile save error", error)
        }
    }

    private static func load(from url: URL) throws -> [SpeakerVerifier.Profile] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([SpeakerVerifier.Profile].self, from: data)
    }
}


