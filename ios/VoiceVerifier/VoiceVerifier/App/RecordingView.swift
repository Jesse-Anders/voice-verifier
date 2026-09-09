import SwiftUI
import Foundation
import UniformTypeIdentifiers

// View to record or import test audio and produce a "concat" WAV containing
// only the segments that verify as the selected profile.
// Pipeline when processing a test file:
// - Fetch segments from the server (Amplitude or SCD depending on Settings)
// - For each segment, request /embed_mean and cosine against training embedding
// - Apply duration-aware threshold; keep accepted segments
// - Concatenate accepted segments with short crossfades and write <name>.concat.wav
// - Save a JSON summary alongside the output file for debugging/analysis
// Notes:
// - SCD boundaries are used as-is (no client smoothing) to preserve change points
// - Amplitude boundaries are smoothed to remove micro blips and add preroll/postroll
// - Processing runs off the main thread and reports progress to the UI
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
    @State private var concatURL: URL?
    @State private var processingInfo: String?
    @State private var showImporter = false
    private enum ImportMode { case test, training }
    @State private var importMode: ImportMode = .test
    @State private var memos: [Memo] = []
    @State private var rmsTimer: Timer? = nil
    @State private var showRenameSheet = false
    @State private var renamingMemo: Memo? = nil
    @State private var newTitle: String = ""

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
            memoList()
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
            if let info = processingInfo { Text(info).font(.footnote).foregroundStyle(.secondary) }
            Text("Only \(profile.name)'s speech will be written to the file.")
                .font(.footnote).foregroundStyle(.secondary)
        }
        .padding()
        .onAppear { setup() }
        .onDisappear { engine?.stop(); player?.stop(); player = nil }
        .sheet(isPresented: $showShare) {
            if let url = fileURL { ActivityView(activityItems: [url]) }
        }
        .sheet(isPresented: $showRenameSheet) {
            VStack(spacing: 16) {
                Text("Rename Memo").font(.headline)
                TextField("Title", text: $newTitle)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Cancel") { showRenameSheet = false }
                    Spacer()
                    Button("Save") {
                        if let m = renamingMemo {
                            MemoStore.shared.rename(profileId: profile.id, memoId: m.id, title: newTitle)
                            memos = MemoStore.shared.load(profileId: profile.id)
                        }
                        showRenameSheet = false
                    }
                }
            }
            .padding()
            .presentationDetents([.height(180)])
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
                        DispatchQueue.main.async { fileURL = dst; savedBytes = bytes; scoreText = "Processing..."; processingInfo = nil }
                        // Build concatenated gated WAV using server segments+scores
                        DispatchQueue.global(qos: .userInitiated).async {
                            if let out = try? buildConcatenatedWav(testURL: dst, profile: profile) {
                                DispatchQueue.main.async {
                                    concatURL = out
                                    scoreText = "Ready"
                                    processingInfo = nil
                                    fileURL = out
                                    savedBytes = (try? FileManager.default.attributesOfItem(atPath: out.path)[.size] as? NSNumber)?.int64Value ?? 0
                                }
                                // Create memo for imported test
                                self.createMemoIfPossible(original: dst, concat: out)
                            } else {
                                DispatchQueue.main.async { scoreText = "Process failed"; processingInfo = nil }
                            }
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

    @ViewBuilder
    private func memoList() -> some View {
        if memos.isEmpty {
            EmptyView()
        } else {
            Divider()
            Text("Memos").font(.headline)
                List {
                    ForEach(memos) { memo in
                        memoRow(for: memo)
                            .listRowInsets(EdgeInsets(top: 4, leading: 0, bottom: 4, trailing: 0))
                    }
                    .onDelete { indexSet in
                        for index in indexSet {
                            let m = memos[index]
                            MemoStore.shared.delete(profileId: profile.id, memo: m)
                        }
                        memos = MemoStore.shared.load(profileId: profile.id)
                    }
                }
                .listStyle(.plain)
                .frame(maxHeight: 220)
        }
    }

    @ViewBuilder
    private func memoRow(for memo: Memo) -> some View {
        HStack {
            NavigationLink(destination: MemoDetailView(memo: memo)) {
                HStack {
                    VStack(alignment: .leading) {
                        let titleText = memo.title?.isEmpty == false ? memo.title! : memo.createdAt.formatted(date: .abbreviated, time: .standard)
                        Text(titleText)
                    }
                    Spacer()
                    Text(URL(fileURLWithPath: memo.concatFile).lastPathComponent)
                        .font(.caption)
                }
            }
        }
        .padding(.vertical, 4)
        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
            Button(role: .destructive) {
                MemoStore.shared.delete(profileId: profile.id, memo: memo)
                memos = MemoStore.shared.load(profileId: profile.id)
            } label: { Label("Delete", systemImage: "trash") }
            Button {
                renamingMemo = memo
                newTitle = memo.title ?? ""
                showRenameSheet = true
            } label: { Label("Rename", systemImage: "pencil") }
        }
    }

    private func setup() {
        guard verifier == nil else { return }
        ModelProvider.shared.loadIfNeeded { v in
            // Local model is optional (server-first path). Proceed even if v == nil.
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
            // Load memos for this profile
            memos = MemoStore.shared.load(profileId: profile.id)
        }
    }

    private func createMemoIfPossible(original: URL, concat: URL) {
        let mode = SegmentsAPI.shared.currentMode()
        var plotURL: URL? = nil
        if let train = trainingURL {
            // Prefer rendering from decisions JSON when available to guarantee 1:1 with audio/logs
            let summary = concat.deletingPathExtension().appendingPathExtension("json")
            let pngData: Data?
            if FileManager.default.fileExists(atPath: summary.path), let decisionsData = try? Data(contentsOf: summary) {
                // Optional meta block for footer
                let cfg = SegmentsAPI.shared.slidingConfig()
                var meta: [String: Any] = [:]
                meta["mode"] = mode.rawValue
                meta["pp"] = cfg?.pp ?? "none"
                if let v = cfg?.win_sec { meta["win_sec"] = v }
                if let v = cfg?.hop_sec { meta["hop_sec"] = v }
                if let v = cfg?.min_keep_sec { meta["min_keep_sec"] = v }
                if let v = cfg?.top_db { meta["top_db"] = v }
                // Tail drill settings for footer (if available)
                if let v = cfg?.tail_drill_step_sec { meta["tail_drill_step_sec"] = v }
                if let v = cfg?.tail_drill_min_len_sec { meta["tail_drill_min_len_sec"] = v }
                if let v = cfg?.tail_drill_max_sec { meta["tail_drill_max_sec"] = v }
                if let v = cfg?.tail_drill_eps { meta["tail_drill_eps"] = v }
                if let v = cfg?.tail_drill_vwin_sec { meta["tail_drill_vwin_sec"] = v }
                let metaData = try? JSONSerialization.data(withJSONObject: meta, options: [])
                pngData = try? PlotAPI.shared.plotFromDecisions(test: original, decisionsJSON: decisionsData, metaJSON: metaData)
            } else {
                // Fallback to server ML plot
                if mode == .sliding {
                    let ppOverride = SegmentsAPI.shared.slidingConfig()?.pp
                    pngData = try? PlotAPI.shared.plotPairSliding(train: train, test: original, pp: ppOverride)
                } else {
                    pngData = try? PlotAPI.shared.plotPair(train: train, test: original)
                }
            }
            if let png = pngData {
                let out = original.deletingPathExtension().appendingPathExtension("plot.png")
                if (try? png.write(to: out)) != nil { plotURL = out }
            }
        }
        let summaryURL = concat.deletingPathExtension().appendingPathExtension("json")
        // Fallback: if plot failed, still create memo using concat path as placeholder (image load will simply omit)
        let plot = plotURL ?? concat
        MemoStore.shared.add(
            profileId: profile.id,
            original: original,
            concat: concat,
            plot: plot,
            summary: FileManager.default.fileExists(atPath: summaryURL.path) ? summaryURL : nil,
            mode: mode.rawValue,
            pp: "none"
        )
        memos = MemoStore.shared.load(profileId: profile.id)
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
            rmsTimer?.invalidate(); rmsTimer = nil; rms = 0
            // Compute saved file size
            if let url = fileURL {
                savedBytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
                print("Recording stopped. Bytes saved: \(savedBytes). Frames=\(engine?.totalFramesWritten() ?? 0)")
                DispatchQueue.main.async { scoreText = "Processing..."; processingInfo = nil }
                // Build concatenated gated WAV using server
                DispatchQueue.global(qos: .userInitiated).async {
                    if let out = try? buildConcatenatedWav(testURL: url, profile: profile) {
                        DispatchQueue.main.async { concatURL = out; scoreText = "Ready"; processingInfo = nil; fileURL = out; savedBytes = (try? FileManager.default.attributesOfItem(atPath: out.path)[.size] as? NSNumber)?.int64Value ?? 0 }
                        // Generate plot and create memo
                        self.createMemoIfPossible(original: url, concat: out)
                    } else {
                        DispatchQueue.main.async { scoreText = "Process failed"; processingInfo = nil }
                    }
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
            rmsTimer?.invalidate()
            rmsTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { _ in
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
    private struct SegmentDecision: Codable {
        let index: Int
        let start: Double
        let end: Double
        let duration: Double
        let score: Float
        let threshold: Float
        let accepted: Bool
    }
    // Build concatenated WAV by calling server /segments and scoring each segment
    private func buildConcatenatedWav(testURL: URL, profile: SpeakerVerifier.Profile) throws -> URL? {
        // Resolve training file path
        guard let tname = profile.trainingFiles?.first else { return nil }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let trainURL = docs.appendingPathComponent(tname)
        guard FileManager.default.fileExists(atPath: trainURL.path) else { return nil }

        // Fetch segments from server based on selected mode
        let mode = SegmentsAPI.shared.currentMode()
        print("RecordingView: segmentation mode=\(mode.rawValue)")
        let rawSegs: [[Double]]
        // Resolve pp for sliding up front so we can use it for both server call and local scoring
        let ppOverride = (mode == .sliding) ? (SegmentsAPI.shared.slidingConfig()?.pp) : nil
        if mode == .sliding {
            rawSegs = try SegmentsAPI.shared.slidingSegments(trainURL: trainURL, testURL: testURL)
        } else {
            rawSegs = try SegmentsAPI.shared.segments(fileURL: testURL, mode: mode)
        }
        if rawSegs.isEmpty { return nil }
        // Sliding: use server segments as-is; Amplitude: apply smoothing
        let segs: [(Double, Double)]
        if mode == .sliding {
            segs = rawSegs.compactMap { $0.count >= 2 ? ($0[0], $0[1]) : nil }
        } else {
            segs = smoothSegments(rawSegs, minGap: 0.2, minLen: 0.25, preroll: 0.08, postroll: 0.08)
        }
        if segs.isEmpty { return nil }

        // Ensure server base URL is refreshed
        EmbedAPI.shared.refreshBaseURL()
        // Get test file duration to clamp end-of-file overhang when computing thresholds
        var testDurationSec: Double? = nil
        if let f = try? AVAudioFile(forReading: testURL) {
            let sr = f.processingFormat.sampleRate
            testDurationSec = Double(f.length) / sr
        }
        var accepted: [(Double, Double)] = []
        // Fetch sliding config to apply tail trim only at concatenation time
        let slidingCfg = (mode == .sliding) ? SegmentsAPI.shared.slidingConfig() : nil
        let tailTrim = (slidingCfg?.tail_trim_sec ?? 0)
        let headTrim = (slidingCfg?.head_trim_sec ?? 0)
        var decisions: [SegmentDecision] = []
        let total = segs.count
        var idx = 0
        // Build slices and batch request to minimize network overhead
        var slicePairs: [[Double]] = []
        var starts: [Double] = []
        var ends: [Double] = []
        for seg in segs {
            let s = max(0.0, seg.0); let e = max(s, seg.1)
            let eEff = testDurationSec != nil ? min(e, testDurationSec!) : e
            let scoreStart = (mode == .sliding) ? min(eEff, s + headTrim) : s
            starts.append(s); ends.append(e)
            slicePairs.append([scoreStart, eEff])
        }
        let results = try EmbedAPI.shared.scoreCosineSliceBatch(trainURL: trainURL, testURL: testURL, slices: slicePairs, pp: ppOverride, log: 0, timeout: 120)
        for i in 0..<results.count {
            idx = i
            let s = starts[i]; let e = ends[i]
            let r = results[i]
            let sc = Float(r.score)
            let thr = Float(r.threshold)
            let ok = r.accepted
            let d = r.duration
            if ok {
                let eEff = slicePairs[i][1]
                let eTrimBase = (mode == .sliding) ? max(s, eEff - tailTrim) : eEff
                let eTrim = testDurationSec != nil ? min(eTrimBase, testDurationSec!) : eTrimBase
                let sTrim = (mode == .sliding) ? min(eTrim, s + headTrim) : s
                if eTrim - sTrim >= 0.05 { accepted.append((sTrim, eTrim)) }
            }
            decisions.append(SegmentDecision(index: i + 1, start: s, end: e, duration: d, score: sc, threshold: thr, accepted: ok))
            print(String(format: "seg %d/%d  dur=%.2fs  score=%.3f  thr=%.3f  %@", i+1, total, d, sc, thr, ok ? "ACCEPT" : "REJECT"))
            DispatchQueue.main.async {
                processingInfo = String(format: "%d/%d  dur=%.2fs  score=%.3f  thr=%.3f  %@", i+1, total, d, sc, thr, ok ? "ACCEPT" : "REJECT")
            }
        }
        if accepted.isEmpty {
            // Write summary JSON for debugging even on full rejection
            let summaryURL = testURL.deletingPathExtension().appendingPathExtension("segments.json")
            if let data = try? JSONEncoder().encode(decisions) { try? data.write(to: summaryURL) }
            return nil
        }
        let out = testURL.deletingPathExtension().appendingPathExtension("concat.wav")
        try concatToFile(source: testURL, intervals: accepted, outURL: out, xfadeMs: 12)
        // Emit JSON summary alongside output
        let summaryURL = out.deletingPathExtension().appendingPathExtension("json")
        if let data = try? JSONEncoder().encode(decisions) { try? data.write(to: summaryURL) }
        return out
    }

    private func durationThreshold(_ d: Double) -> Float {
        // Smooth saturating curve (no kink), aligned with tools
        let tMin = 0.06
        let tMax = 0.55
        let tau  = 2.8
        let dd = max(0.0, d)
        let val = tMin + (tMax - tMin) * (1.0 - exp(-dd / tau))
        return Float(val)
    }

    // Merge tiny gaps, drop very short segments, add small pre/post roll
    private func smoothSegments(_ segments: [[Double]], minGap: Double, minLen: Double, preroll: Double, postroll: Double) -> [(Double, Double)] {
        var segs: [(Double, Double)] = []
        for s in segments {
            if s.count >= 2 { segs.append((s[0], s[1])) }
        }
        if segs.isEmpty { return [] }
        segs.sort { $0.0 < $1.0 }
        var merged: [(Double, Double)] = []
        var cs = segs[0].0
        var ce = segs[0].1
        for i in 1..<segs.count {
            let (s, e) = segs[i]
            if s - ce < minGap {
                ce = max(ce, e)
            } else {
                if (ce - cs) >= minLen { merged.append((max(0.0, cs - preroll), max(0.0, ce + postroll))) }
                cs = s; ce = e
            }
        }
        if (ce - cs) >= minLen { merged.append((max(0.0, cs - preroll), max(0.0, ce + postroll))) }
        return merged
    }

    // Write [start,end] seconds of url to a temporary WAV and return its URL
    private func sliceWav(from url: URL, start: Double, end: Double) throws -> URL {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let sr = format.sampleRate
        let sFrame = AVAudioFramePosition(max(0, Int64(start * sr)))
        let eFrame = AVAudioFramePosition(max(sFrame, Int64(end * sr)))
        let frames = AVAudioFrameCount(max(0, eFrame - sFrame))
        file.framePosition = sFrame
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw NSError(domain: "SliceWav", code: -1, userInfo: [NSLocalizedDescriptionKey: "buffer alloc failed"])
        }
        try file.read(into: buf, frameCount: frames)
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("seg_\(UUID().uuidString).wav")
        let out = try AVAudioFile(forWriting: tmp, settings: format.settings)
        try out.write(from: buf)
        return tmp
    }

    private func concatToFile(source: URL, intervals: [(Double, Double)], outURL: URL, xfadeMs: Double) throws {
        let f = try AVAudioFile(forReading: source)
        let fmt = f.processingFormat
        let sr = fmt.sampleRate
        let ch = Int(fmt.channelCount)
        let xfade = AVAudioFrameCount(max(0, Int(xfadeMs * 1e-3 * sr)))
        var buffers: [AVAudioPCMBuffer] = []
        for (s, e) in intervals {
            let sFrame = AVAudioFramePosition(max(0, Int64(s * sr)))
            let eFrame = AVAudioFramePosition(max(sFrame, Int64(e * sr)))
            let n = AVAudioFrameCount(max(0, eFrame - sFrame))
            f.framePosition = sFrame
            guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n) else { continue }
            try f.read(into: buf, frameCount: n)
            buffers.append(buf)
        }
        let out = try AVAudioFile(forWriting: outURL, settings: fmt.settings)
        var carry: AVAudioPCMBuffer? = nil
        for buf in buffers {
            if carry == nil {
                carry = buf
                continue
            }
            guard let prev = carry else { continue }
            let n = min(xfade, prev.frameLength, buf.frameLength)
            // 1) Write prev except its last n frames
            if n < prev.frameLength {
                let keepFrames = prev.frameLength - n
                let tmp = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: keepFrames)!
                tmp.frameLength = keepFrames
                if let src = prev.floatChannelData, let dst = tmp.floatChannelData {
                    for c in 0..<ch {
                        let s = src[c]
                        let d = dst[c]
                        for i in 0..<Int(keepFrames) { d[i] = s[i] }
                    }
                }
                try out.write(from: tmp)
            }
            // 2) Write crossfade mix of length n
            if n > 0 {
                let mix = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
                mix.frameLength = n
                if let psrc = prev.floatChannelData, let csrc = buf.floatChannelData, let mdst = mix.floatChannelData {
                    for c in 0..<ch {
                        let p = psrc[c]
                        let cc = csrc[c]
                        let m = mdst[c]
                        for i in 0..<Int(n) {
                            let fo = 1.0 - Float(i) / Float(n)
                            let fi = 1.0 - fo
                            let idxp = Int(prev.frameLength - n + AVAudioFrameCount(i))
                            m[i] = p[idxp] * fo + cc[i] * fi
                        }
                    }
                }
                try out.write(from: mix)
            }
            // 3) Prepare new carry = current without its first n frames
            let remain = buf.frameLength > n ? (buf.frameLength - n) : 0
            let nextCarry = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: remain)!
            nextCarry.frameLength = remain
            if remain > 0, let csrc = buf.floatChannelData, let ndst = nextCarry.floatChannelData {
                for c in 0..<ch {
                    let cc = csrc[c]; let d = ndst[c]
                    for i in 0..<Int(remain) { d[i] = cc[i + Int(n)] }
                }
            }
            carry = nextCarry
        }
        if let last = carry, last.frameLength > 0 { try out.write(from: last) }
    }
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
            _ = fmt.sampleRate
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




