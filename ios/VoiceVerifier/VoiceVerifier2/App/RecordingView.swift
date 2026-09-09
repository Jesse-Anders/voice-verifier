import SwiftUI
import UniformTypeIdentifiers

struct RecordingView: View {
    @EnvironmentObject var store: ProfileStore
    let profile: SpeakerVerifier.Profile
    @State private var isRecording = false
    @State private var rms: Float = 0
    @State private var verifier: SpeakerVerifier?
    @State private var engine: AudioEngineService?
    @State private var fileURL: URL?
    @State private var savedBytes: Int64 = 0
    @State private var showShare = false
    @State private var player: AVAudioPlayer?
    @State private var scoreText: String?
    @State private var trainingURL: URL?
    @State private var showImporter = false
    private enum ImportMode { case test, training }
    @State private var importMode: ImportMode = .test

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(profile.name).font(.title2).bold()
                Spacer()
                Text(String(format: "thr %.2f", profile.threshold)).foregroundStyle(.secondary)
            }
            HStack {
                Circle().fill(.blue.opacity(Double(min(1, max(0, rms * 10)))))
                    .frame(width: 16, height: 16)
                Text("RMS: \(String(format: "%.3f", rms))")
                Spacer()
            }
            Button(isRecording ? "Stop" : "Record") { toggle() }
                .buttonStyle(.borderedProminent)
            if let url = fileURL {
                VStack(spacing: 8) {
                    Text(url.lastPathComponent).font(.footnote).lineLimit(1)
                    if !isRecording && savedBytes > 0 {
                        Text(String(format: "Saved %.1f MB", Double(savedBytes) / (1024*1024))).font(.footnote).foregroundStyle(.secondary)
                        if let s = scoreText { Text(s).font(.footnote).foregroundStyle(.secondary) }
                        HStack(spacing: 12) {
                            Button(player?.isPlaying == true ? "Stop" : "Play") { togglePlay(url) }
                            Button("Share") { showShare = true }
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            Spacer()
            // Training file controls (if present)
            if let turl = trainingURL {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Training Audio").font(.headline)
                    HStack(spacing: 12) {
                        Text(turl.lastPathComponent).font(.footnote).lineLimit(1)
                        Spacer()
                        Button(player?.isPlaying == true ? "Stop" : "Play") { togglePlay(turl) }
                        Button("Retrain Model") { retrainFromTraining(turl) }
                        Button("Share") { showShare = true; fileURL = turl }
                    }
                    .buttonStyle(.bordered)
                }
            }
            HStack(spacing: 12) {
                Button("Import Test File") { importMode = .test; showImporter = true }
                if trainingURL != nil { Button("Import Training File") { importMode = .training; showImporter = true } }
            }
            Text("Only \(profile.name)'s speech will be written to the file.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .padding()
        .onAppear { setup() }
        .onDisappear { engine?.stop(); player?.stop(); player = nil }
        .sheet(isPresented: $showShare) {
            if let url = fileURL { ActivityView(activityItems: [url]) }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [UTType.audio], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let src = urls.first else { return }
                do {
                    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                    let prefix = (importMode == .training) ? "enroll" : "test"
                    let dst = docs.appendingPathComponent("\(prefix)_\(Int(Date().timeIntervalSince1970))_\(src.lastPathComponent)")
                    if FileManager.default.fileExists(atPath: dst.path) { try? FileManager.default.removeItem(at: dst) }
                    _ = src.startAccessingSecurityScopedResource()
                    defer { src.stopAccessingSecurityScopedResource() }
                    try FileManager.default.copyItem(at: src, to: dst)
                    if importMode == .training {
                        trainingURL = dst
                        retrainFromTraining(dst)
                    } else {
                        // Update UI to show imported test file and size
                        let bytes = (try? FileManager.default.attributesOfItem(atPath: dst.path)[.size] as? NSNumber)?.int64Value ?? 0
                        DispatchQueue.main.async { fileURL = dst; savedBytes = bytes; scoreText = "Scoring..." }
                        DispatchQueue.global(qos: .userInitiated).async {
                            let result = computeScore(url: dst, profile: profile)
                            DispatchQueue.main.async { scoreText = result }
                        }
                    }
                } catch {
                    print("Import error: \(error)")
                }
            case .failure(let e):
                print("Import error: \(e)")
            }
        }
    }

    private func setup() {
        guard verifier == nil else { return }
        ModelProvider.shared.loadIfNeeded { v in
            guard let v = v else { print("Model load failed"); return }
            if let eng = AudioEngineService(verifier: v) {
                verifier = v
                engine = eng
            } else {
                print("AudioEngine init failed")
            }
            // Resolve training file path if stored on profile
            if let name = profile.trainingFiles?.first {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let url = docs.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: url.path) { trainingURL = url }
            }
        }
    }

    private func toggle() {
        // Ensure engine exists even if model is still loading
        if engine == nil {
            if let eng = AudioEngineService(verifier: ModelProvider.shared.verifier) {
                engine = eng
            } else {
                print("Recording: could not init AudioEngineService")
                return
            }
        }
        guard let eng = engine else { return }
        if isRecording {
            eng.stop(); isRecording = false
            // Compute saved file size
            if let url = fileURL {
                savedBytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
                print("Recording stopped. Bytes saved: \(savedBytes). Frames=\(engine?.totalFramesWritten() ?? 0)")
                // Compute ML score of the memo
                DispatchQueue.global(qos: .userInitiated).async {
                    let result = computeScore(url: url, profile: profile)
                    DispatchQueue.main.async { scoreText = result }
                }
            }
        } else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let url = docs.appendingPathComponent("memo_\(Int(Date().timeIntervalSince1970)).wav")
            fileURL = url
            // Request mic permission non-blocking, then start
            AudioEngineService.requestMicrophonePermission { ok in
                guard ok else { print("Microphone permission denied"); return }
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try eng.configure(mode: .record(profile: profile), outputURL: url)
                        try eng.start()
                        DispatchQueue.main.async {
                            isRecording = true
                            savedBytes = 0
                            scoreText = nil
                        }
                    } catch {
                        print("AudioEngine start error: \(error)")
                    }
                }
            }
            Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { _ in
                rms = eng.levelRMS
            }
        }
    }
}

import UIKit
struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

import AVFoundation
extension RecordingView {
    private func togglePlay(_ url: URL) {
        if let p = player, p.isPlaying {
            p.stop(); player = nil
        } else {
            do {
                let p = try AVAudioPlayer(contentsOf: url)
                p.prepareToPlay()
                p.play()
                player = p
            } catch {
                print("Player error: \(error)")
            }
        }
    }

    private func retrainFromTraining(_ url: URL) {
        guard let v = verifier else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let f = try AVAudioFile(forReading: url)
                let fmt = f.processingFormat
                let frames = AVAudioFrameCount(f.length)
                guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) else { return }
                try f.read(into: buf)
                let sr = fmt.sampleRate
                let ch = Int(fmt.channelCount)
                var samples: [Float] = []
                if let data = buf.floatChannelData {
                    let n = Int(buf.frameLength)
                    if ch == 1 {
                        samples = Array(UnsafeBufferPointer(start: data[0], count: n))
                    } else {
                        samples = [Float](repeating: 0, count: n)
                        for i in 0..<n { samples[i] = 0.5 * (data[0][i] + data[1][i]) }
                    }
                } else if let dataI16 = buf.int16ChannelData {
                    let n = Int(buf.frameLength)
                    if ch == 1 {
                        samples = (0..<n).map { Float(dataI16[0][$0]) / 32768.0 }
                    } else {
                        samples = [Float](repeating: 0, count: n)
                        for i in 0..<n { samples[i] = 0.5 * (Float(dataI16[0][i]) + Float(dataI16[1][i])) / 32768.0 }
                    }
                }
                if let centroid = v.computeCentroid(samples: samples, inputSampleRate: sr) {
                    let updated = SpeakerVerifier.Profile(id: profile.id, name: profile.name, centroid: centroid, threshold: profile.threshold, trainingFiles: profile.trainingFiles)
                    DispatchQueue.main.async { store.update(profile: updated) }
                }
            } catch {
                print("Retrain error: \(error)")
            }
        }
    }

    // Load samples from WAV and compute speaker score across entire file
    private func computeScore(url: URL, profile: SpeakerVerifier.Profile) -> String {
        do {
            let f = try AVAudioFile(forReading: url)
            let fmt = f.processingFormat
            let frames = AVAudioFrameCount(f.length)
            guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames) else { return "Score: load error" }
            try f.read(into: buf)
            let sr = fmt.sampleRate
            let ch = Int(fmt.channelCount)
            var samples: [Float] = []
            if let data = buf.floatChannelData {
                let n = Int(buf.frameLength)
                if ch == 1 {
                    samples = Array(UnsafeBufferPointer(start: data[0], count: n))
                } else {
                    samples = [Float](repeating: 0, count: n)
                    for i in 0..<n { samples[i] = 0.5 * (data[0][i] + data[1][i]) }
                }
            } else if let dataI16 = buf.int16ChannelData {
                let n = Int(buf.frameLength)
                if ch == 1 {
                    samples = (0..<n).map { Float(dataI16[0][$0]) / 32768.0 }
                } else {
                    samples = [Float](repeating: 0, count: n)
                    for i in 0..<n {
                        samples[i] = 0.5 * (Float(dataI16[0][i]) + Float(dataI16[1][i])) / 32768.0
                    }
                }
            } else {
                return "Score: unsupported format"
            }
            guard !profile.centroid.isEmpty else { return "Score: no centroid" }
            // Option A (disabled by request): do not use server pairwise scoring when centroid is present
            // Option B: fetch remote embedding and cosine locally against centroid (no local model required)
            if let remoteEmb = try? EmbedAPI.shared.embedMean(fileURL: url) {
                let s = cosineSimilarity(remoteEmb, profile.centroid)
                return String(format: "Score: %.3f", s)
            }
            return "Score: network error"
        } catch {
            return "Score error: \(error.localizedDescription)"
        }
    }

    // Minimal local cosine similarity helper (to avoid needing local model)
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        if a.isEmpty || b.isEmpty || a.count != b.count { return 0 }
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0..<a.count {
            let x = Double(a[i])
            let y = Double(b[i])
            dot += x * y
            na += x * x
            nb += y * y
        }
        let denom = (sqrt(na) * sqrt(nb)) + 1e-12
        return Float(dot / denom)
    }

    private func trimSilence(_ x: [Float], sr: Double, topDb: Float = 25, frameMs: Double = 30, hopMs: Double = 10) -> [Float] {
        if x.isEmpty { return x }
        let frame = max(1, Int(frameMs * sr / 1000.0))
        let hop = max(1, Int(hopMs * sr / 1000.0))
        var rmsDb: [Float] = []
        var idx = 0
        while idx < x.count {
            let end = min(x.count, idx + frame)
            var sum: Float = 0
            for i in idx..<end { sum += x[i] * x[i] }
            let r = sqrt(sum / Float(end - idx) + 1e-12)
            let db = 20.0 * log10f(max(r, 1e-6))
            rmsDb.append(db)
            idx += hop
        }
        guard let maxDb = rmsDb.max() else { return x }
        let thr = maxDb - topDb
        // Find segments where db > thr
        var segments: [(Int, Int)] = []
        var inSeg = false
        var segStart = 0
        for (fi, db) in rmsDb.enumerated() {
            if db > thr {
                if !inSeg { inSeg = true; segStart = fi }
            } else {
                if inSeg {
                    let startSample = segStart * hop
                    let endSample = min(x.count, fi * hop + frame)
                    segments.append((startSample, endSample))
                    inSeg = false
                }
            }
        }
        if inSeg {
            let startSample = segStart * hop
            let endSample = x.count
            segments.append((startSample, endSample))
        }
        if segments.isEmpty { return x }
        // Concatenate kept segments
        var out: [Float] = []
        for (s,e) in segments { out += x[s..<e] }
        return out
    }

    // Resample utility to 16 kHz using AVAudioConverter
    private func resampleTo16k(_ input: [Float], fromSR: Double) -> [Float] {
        if input.isEmpty || abs(fromSR - 16000.0) < 1.0 { return input }
        let src = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: fromSR, channels: 1, interleaved: false)!
        let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000.0, channels: 1, interleaved: false)!
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(input.count)) else { return input }
        inBuf.frameLength = AVAudioFrameCount(input.count)
        if let ptr = inBuf.floatChannelData?[0] { for i in 0..<input.count { ptr[i] = input[i] } }
        let outFrames = AVAudioFrameCount(Double(input.count) * (16000.0 / fromSR) + 16)
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outFrames) else { return input }
        let conv = AVAudioConverter(from: src, to: dst)
        var err: NSError?
        let block: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return inBuf
        }
        conv?.convert(to: outBuf, error: &err, withInputFrom: block)
        if err != nil { return input }
        let frames = Int(outBuf.frameLength)
        guard let outPtr = outBuf.floatChannelData?[0] else { return input }
        return Array(UnsafeBufferPointer(start: outPtr, count: frames))
    }

    private func l2normalize(_ v: [Float]) -> [Float] {
        let norm = sqrt(v.reduce(Double(0)) { $0 + Double($1 * $1) })
        if norm <= 1e-12 { return v }
        let inv = 1.0 / norm
        return v.map { $0 * Float(inv) }
    }
}
// Removed inline player; basic UI restored


