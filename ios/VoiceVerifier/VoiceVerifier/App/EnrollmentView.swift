import SwiftUI
import UniformTypeIdentifiers
import AVFoundation

// View to create a new speaker profile
// Flow:
// 1) User records a sample (or imports an existing training file)
// 2) We generate a preview WAV and request a server embedding to create a centroid
// 3) On success, Save creates a profile with name, centroid and training file path
// Notes:
// - We keep UI responsive by running embedding/training off the main thread
// - We require server-side embedding to match the runtime pipeline
// - RMS dot shows live input level; seconds increments while recording
struct EnrollmentView: View {
    @EnvironmentObject var store: ProfileStore
    @State private var name: String = ""
    @State private var seconds: Int = 0
    @State private var isRecording = false
    @State private var rms: Float = 0
    @State private var verifier: SpeakerVerifier?
    @State private var engine: AudioEngineService?
    @State private var previewURL: URL?
    @State private var previewPlayer: AVAudioPlayer?
    @State private var isTrained = false
    @State private var capturedSamples: [Float] = []
    @State private var trainedCentroid: [Float]? = nil
    @State private var showImportTraining = false
    @State private var showShare = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            TextField("Name (e.g., John)", text: $name)
                .textFieldStyle(.roundedBorder)

            // Live level indicator and elapsed seconds while recording
            HStack {
                Circle().fill(.green.opacity(Double(min(1, max(0, rms * 10)))))
                    .frame(width: 16, height: 16)
                Text("RMS: \(String(format: "%.3f", rms))")
                Spacer()
                Text("\(seconds)s")
            }

            // Primary actions: record/stop, kick off training, import, save
            HStack {
                Button(isRecording ? "Stop" : "Start") { toggle() }
                    .buttonStyle(.borderedProminent)
                Button("Train Model") { trainProfile() }
                    .disabled(isRecording || (capturedSamples.isEmpty && previewURL == nil) || verifier == nil)
                Button("Import Training File") { showImportTraining = true }
                Button("Save") { saveProfile() }
                    .disabled(!canSave)
            }
            if let url = previewURL {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 12) {
                        Text(url.lastPathComponent).font(.footnote)
                        Button(previewPlayer?.isPlaying == true ? "Stop" : "Play") { togglePlay(url) }
                        Button("Share") { showShare = true }
                            .buttonStyle(.bordered)
                    }
                    if !isTrained {
                        Text("Training… please wait before saving")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Text("Record 60–120 seconds of your normal speech.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .onAppear { setup() }
        .onDisappear { engine?.stop() }
        .sheet(isPresented: $showShare) {
            if let url = previewURL { ActivityView(activityItems: [url]) }
        }
        // Import an external training file into Documents and compute centroid
        .fileImporter(isPresented: $showImportTraining, allowedContentTypes: [.audio], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let src = urls.first else { return }
                do {
                    // Copy into Documents for persistence
                    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                    let dst = docs.appendingPathComponent("enroll_\(Int(Date().timeIntervalSince1970))_\(src.lastPathComponent)")
                    if FileManager.default.fileExists(atPath: dst.path) { try? FileManager.default.removeItem(at: dst) }
                    _ = src.startAccessingSecurityScopedResource()
                    defer { src.stopAccessingSecurityScopedResource() }
                    do {
                        try FileManager.default.copyItem(at: src, to: dst)
                    } catch {
                        // Fallback: read bytes and write if direct copy is not permitted by provider
                        let nsError = error as NSError
                        if nsError.domain == NSCocoaErrorDomain && nsError.code == 257 {
                            let data = try Data(contentsOf: src)
                            try data.write(to: dst, options: .atomic)
                        } else {
                            throw error
                        }
                    }
                    previewURL = dst
                    // Compute centroid from server embedding (no local fallback) - do not block UI
                    DispatchQueue.global(qos: .userInitiated).async {
                        if let c = try? EmbedAPI.shared.embedMean(fileURL: dst) {
                            DispatchQueue.main.async { trainedCentroid = c; isTrained = true }
                        } else {
                            DispatchQueue.main.async {
                                isTrained = false
                                print("Enrollment: server centroid failed; no fallback to local to avoid pipeline mismatch")
                            }
                        }
                    }
                } catch {
                    print("Import training error: \(error)")
                }
            case .failure(let e):
                print("Import canceled/error: \(e)")
            }
        }
    }

    private var canSave: Bool {
        let hasName = !name.trimmingCharacters(in: .whitespaces).isEmpty
        return hasName && isTrained && (trainedCentroid != nil)
    }

    private func setup() {
        guard verifier == nil else { return }
        ModelProvider.shared.loadIfNeeded { v in
            if let eng = AudioEngineService(verifier: v) {
                verifier = v
                engine = eng
                try? eng.configure(mode: .enroll, outputURL: nil)
                Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                    if isRecording { seconds += 1 }
                    rms = eng.levelRMS
                }
            } else {
                print("AudioEngine init failed")
            }
        }
    }

    private func toggle() {
        guard let eng = engine else { return }
        if isRecording {
            eng.stop(); isRecording = false
            // On stop, capture samples and write a preview file
            let samples = eng.copyAndResetSamples()
            capturedSamples = samples
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let url = docs.appendingPathComponent("enroll_\(Int(Date().timeIntervalSince1970)).wav")
            do {
                try writePreviewWav(samples: samples, sampleRate: AVAudioSession.sharedInstance().sampleRate, url: url)
                previewURL = url
                // Immediately compute server centroid from the newly recorded preview file (non-blocking)
                DispatchQueue.global(qos: .userInitiated).async {
                    if let c = try? EmbedAPI.shared.embedMean(fileURL: url) {
                        DispatchQueue.main.async { trainedCentroid = c; isTrained = true }
                    } else {
                        DispatchQueue.main.async {
                            isTrained = false
                            print("Enrollment: server centroid failed; no fallback to local to avoid pipeline mismatch")
                        }
                    }
                }
            } catch {
                print("Preview write error: \(error)")
                previewURL = nil
            }
        } else {
            do { try eng.start(); isRecording = true } catch { isRecording = false; print("AudioEngine start error: \(error)") }
        }
    }

    private func saveProfile() {
        guard let centroid = trainedCentroid, isTrained else { return }
        let files = previewURL != nil ? [previewURL!.lastPathComponent] : nil
        store.add(name: name, centroid: centroid, trainingFiles: files)
        dismiss()
    }

    // Optional manual retrain button to recompute centroid
    private func trainProfile() {
        // Prefer server centroid when previewURL exists; fallback to local captured samples
        if let url = previewURL {
            DispatchQueue.global(qos: .userInitiated).async {
                if let c = try? EmbedAPI.shared.embedMean(fileURL: url) {
                    DispatchQueue.main.async { trainedCentroid = c; isTrained = true }
                } else {
                    DispatchQueue.main.async { isTrained = false }
                }
            }
            return
        }
        guard let v = verifier else { isTrained = false; return }
        let sr = AVAudioSession.sharedInstance().sampleRate
        if let c = v.computeCentroid(samples: capturedSamples, inputSampleRate: sr) {
            trainedCentroid = c
            isTrained = true
        } else {
            isTrained = false
        }
    }

    private func writePreviewWav(samples: [Float], sampleRate: Double, url: URL) throws {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        if FileManager.default.fileExists(atPath: url.path) { try? FileManager.default.removeItem(at: url) }
        let f = try AVAudioFile(forWriting: url, settings: fmt.settings)
        let n = samples.count
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n)) else { return }
        buf.frameLength = AVAudioFrameCount(n)
        let dst = buf.floatChannelData![0]
        for i in 0..<n { dst[i] = samples[i] }
        try f.write(from: buf)
    }

    private func loadSamples(from url: URL) throws -> ([Float], Double) {
        let f = try AVAudioFile(forReading: url)
        let fmt = f.processingFormat
        let frames = AVAudioFrameCount(f.length)
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) else { return ([], 16000.0) }
        try f.read(into: buf)
        let sr = fmt.sampleRate
        let ch = Int(fmt.channelCount)
        var samples: [Float] = []
        if let data = buf.floatChannelData {
            let n = Int(buf.frameLength)
            if ch == 1 { samples = Array(UnsafeBufferPointer(start: data[0], count: n)) }
            else {
                samples = [Float](repeating: 0, count: n)
                for i in 0..<n { samples[i] = 0.5 * (data[0][i] + data[1][i]) }
            }
        } else if let dataI16 = buf.int16ChannelData {
            let n = Int(buf.frameLength)
            if ch == 1 { samples = (0..<n).map { Float(dataI16[0][$0]) / 32768.0 } }
            else {
                samples = [Float](repeating: 0, count: n)
                for i in 0..<n { samples[i] = 0.5 * (Float(dataI16[0][i]) + Float(dataI16[1][i])) / 32768.0 }
            }
        }
        return (samples, sr)
    }

    private func togglePlay(_ url: URL) {
        if let p = previewPlayer, p.isPlaying { p.stop(); previewPlayer = nil }
        else {
            do { let p = try AVAudioPlayer(contentsOf: url); p.prepareToPlay(); p.play(); previewPlayer = p } catch { print("Enroll play error: \(error)") }
        }
    }
}


