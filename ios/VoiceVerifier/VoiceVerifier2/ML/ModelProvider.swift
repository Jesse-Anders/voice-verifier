import Foundation

final class ModelProvider: ObservableObject {
    static let shared = ModelProvider()
    private init() {}

    @Published var verifier: SpeakerVerifier?
    @Published var isLoading: Bool = false

    func loadIfNeeded(completion: @escaping (SpeakerVerifier?) -> Void) {
        if let v = verifier { completion(v); return }
        if isLoading { DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in completion(self?.verifier) }; return }

        // Option to skip local model entirely (use server embeddings only)
        if UserDefaults.standard.bool(forKey: "SkipLocalModel") == true {
            completion(nil)
            return
        }

        isLoading = true
        // Use raw embedding model only (scorer uses ops not available on-device)
        let urlPT = Bundle.main.url(forResource: "ecapa_embedding", withExtension: "pt")
        let urlPTL = Bundle.main.url(forResource: "ecapa_embedding", withExtension: "ptl")
        guard let modelURL = urlPT ?? urlPTL else {
            DispatchQueue.main.async { [weak self] in self?.isLoading = false; completion(nil) }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let v = SpeakerVerifier(modelURL: modelURL)
            DispatchQueue.main.async {
                self?.verifier = v
                self?.isLoading = false
                completion(v)
            }
        }
    }
}


