import Foundation
import AVFoundation

// Minimal on-device verifier used for local embeddings and experiments.
// In the current app, server embeddings are preferred; this class remains
// useful for offline testing and to compute centroids locally if needed.

final class SpeakerVerifier {
    struct Profile: Codable, Identifiable, Equatable {
        let id: UUID
        var name: String
        var centroid: [Float]
        var threshold: Float
        var trainingFiles: [String]? // filenames in Documents
    }

    private let model: TorchModule
    private let sampleRate: Double = 16000.0
    private let windowSeconds: Double = 3.0
    private let strideSeconds: Double = 1.0
    private let pythonExact: Bool = true
    private let debugLogs: Bool = true

    init?(modelURL: URL) {
        guard let tm = TorchModule(file: modelURL.path) else { return nil }
        self.model = tm
    }

    // Downsample to 16 kHz (AVAudioConverter) if needed before embedding
    func embed(samples: [Float], inputSampleRate: Double = 16000.0) -> [Float]? {
        // Plain embedding path (resample then forward)
        let mono16k = abs(inputSampleRate - sampleRate) < 1 ? samples : resampleAV(input: samples, fromSR: inputSampleRate, toSR: sampleRate)
        let ns = mono16k as [NSNumber]
        return model.embedAudioPCM(ns)?.map { $0.floatValue }
    }

    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let denom = (l2(a) * l2(b)) + 1e-12
        guard denom > 0 else { return 0 }
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        return dot / Float(denom)
    }

    private func l2(_ v: [Float]) -> Double {
        sqrt(v.reduce(Double(0)) { $0 + Double($1 * $1) })
    }

    // Compute the average embedding over 3 s windows with 1 s hop, after a
    // simple VAD trim. Mirrors the Python path closely when pythonExact=true.
    func computeCentroid(samples: [Float], inputSampleRate: Double = 16000.0) -> [Float]? {
        // Resample to 16k and trim silence before windowing, matching Python
        let mono16k = abs(inputSampleRate - sampleRate) < 1 ? samples : resampleAV(input: samples, fromSR: inputSampleRate, toSR: sampleRate)
        let trimmed = trimSilence(mono16k, sr: sampleRate, topDb: pythonExact ? 25 : 35)
        let frames = windowFrames(samples: trimmed)
        let gated = pythonExact ? frames : frames.filter { windowRMS($0) >= 0.02 }
        var embs: [[Float]] = []
        for f in gated.isEmpty ? frames : gated {
            let input = pythonExact ? f : standardize(f)
            if let e = embed(samples: input, inputSampleRate: sampleRate) {
                let vec = pythonExact ? e : l2normalize(e)
                embs.append(vec)
            }
        }
        guard !embs.isEmpty else { return nil }
        let dim = embs[0].count
        var sum = [Float](repeating: 0, count: dim)
        for e in embs { for i in 0..<dim { sum[i] += e[i] } }
        let mean = sum.map { $0 / Float(embs.count) }
        if debugLogs {
            let mnorm = l2(mean)
            let preview = mean.prefix(4).map { String(format: "%.3f", $0) }.joined(separator: ", ")
            print("Centroid: frames=\(embs.count) L2=\(String(format: "%.3f", mnorm)) first4=[\(preview)]")
        }
        return mean
    }

    // Convenience helper: embed and compare against a centroid using cosine.
    func verify(samples: [Float], centroid: [Float], threshold: Float, inputSampleRate: Double = 16000.0) -> (Bool, Float) {
        guard let e = embed(samples: samples, inputSampleRate: inputSampleRate) else { return (false, 0) }
        let score = cosineSimilarity(e, centroid)
        return (score >= threshold, score)
    }

    // Slice samples into overlapping windows for batch embedding.
    func windowFrames(samples: [Float]) -> [[Float]] {
        let win = Int(windowSeconds * sampleRate)
        let stride = Int(strideSeconds * sampleRate)
        if samples.count <= win {
            var padded = samples
            if samples.count < win { padded += [Float](repeating: 0, count: win - samples.count) }
            return [padded]
        }
        var frames: [[Float]] = []
        var start = 0
        while start + win <= samples.count {
            frames.append(Array(samples[start..<(start+win)]))
            start += stride
        }
        return frames
    }

    private func l2normalize(_ v: [Float]) -> [Float] {
        let norm = sqrt(v.reduce(Double(0)) { $0 + Double($1 * $1) })
        if norm <= 1e-12 { return v }
        let inv = 1.0 / norm
        return v.map { $0 * Float(inv) }
    }

    // Energy-based VAD used only for local training convenience.
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
        var segments: [(Int, Int)] = []
        var inSeg = false
        var segStart = 0
        for (fi, db) in rmsDb.enumerated() {
            if db > thr { if !inSeg { inSeg = true; segStart = fi } }
            else if inSeg {
                let startSample = segStart * hop
                let endSample = min(x.count, fi * hop + frame)
                segments.append((startSample, endSample))
                inSeg = false
            }
        }
        if inSeg { segments.append((segStart * hop, x.count)) }
        if segments.isEmpty { return x }
        var out: [Float] = []
        for (s,e) in segments { out += x[s..<e] }
        return out
    }

    // Standardize a frame: zero-mean, unit-variance (if variance > 0)
    private func standardize(_ v: [Float]) -> [Float] {
        if v.isEmpty { return v }
        let mean = v.reduce(Float(0)) { $0 + $1 } / Float(v.count)
        var varSum = Double(0)
        for x in v {
            let d = Double(x - mean)
            varSum += d * d
        }
        let variance = varSum / Double(max(1, v.count - 1))
        let std = sqrt(variance)
        if std < 1e-8 { return v.map { $0 - mean } }
        let inv = 1.0 / std
        return v.map { ($0 - mean) * Float(inv) }
    }

    // Root-mean-square amplitude of a window
    private func windowRMS(_ v: [Float]) -> Float {
        if v.isEmpty { return 0 }
        let sum = v.reduce(Float(0)) { $0 + $1*$1 }
        return sqrt(sum / Float(v.count))
    }

    // resampleLinear removed; using AVAudioConverter-based resampleAV

    private func resampleAV(input: [Float], fromSR: Double, toSR: Double) -> [Float] {
        if input.isEmpty || fromSR == toSR { return input }
        let src = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: fromSR, channels: 1, interleaved: false)!
        let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: toSR, channels: 1, interleaved: false)!
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(input.count)) else { return input }
        inBuf.frameLength = AVAudioFrameCount(input.count)
        let inPtr = inBuf.floatChannelData![0]
        for i in 0..<input.count { inPtr[i] = input[i] }
        let outFrames = AVAudioFrameCount(Double(input.count) * (toSR / fromSR) + 16)
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
}


