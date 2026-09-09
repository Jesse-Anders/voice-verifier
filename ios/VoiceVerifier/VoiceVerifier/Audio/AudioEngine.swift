import Foundation
import AVFoundation
import AVFAudio

// Thin wrapper around AVAudioEngine to capture microphone audio to memory and/or file.
// Responsibilities:
// - Manage microphone permission, session category, and engine lifecycle
// - Convert input to a single-channel float32 stream at the device/native sample rate
// - Publish lightweight state (isRunning, levelRMS) for UI updates
// - In enroll mode: keep a rolling buffer of recent samples for preview/centroid
// - In record mode: stream samples to a WAV file on disk

final class AudioEngineService: ObservableObject {
    enum Mode { case enroll, record(profile: SpeakerVerifier.Profile) }

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let inputBus: AVAudioNodeBus = 0
    // Use input/native sample rate (e.g., 48000) and write directly
    private var targetFormat: AVAudioFormat!

    @Published var isRunning = false
    @Published var levelRMS: Float = 0

    private var pendingSamples: [Float] = []
    private var outputFile: AVAudioFile?
    private var pendingOutputURL: URL?
    private var verifier: SpeakerVerifier?
    private var mode: Mode = .enroll
    private var debugTotalFramesWritten: Int64 = 0

    init?(verifier: SpeakerVerifier?) {
        self.verifier = verifier
        // No blocking permission requests here; handled asynchronously by caller just-in-time
    }

    // Non-blocking permission request for callers to use before starting
    static func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        if #available(iOS 17.0, *) {
            let app = AVAudioApplication.shared
            switch app.recordPermission {
            case .granted: completion(true)
            case .denied: completion(false)
            case .undetermined:
                AVAudioApplication.requestRecordPermission { ok in
                    DispatchQueue.main.async { completion(ok) }
                }
            @unknown default:
                completion(false)
            }
        } else {
            let session = AVAudioSession.sharedInstance()
            switch session.recordPermission {
            case .granted: completion(true)
            case .denied: completion(false)
            case .undetermined:
                session.requestRecordPermission { ok in
                    DispatchQueue.main.async { completion(ok) }
                }
            @unknown default:
                completion(false)
            }
        }
    }

    // Prepare engine state for a given workflow. File is opened later in start().
    func configure(mode: Mode, outputURL: URL?) throws {
        self.mode = mode
        self.pendingSamples.removeAll()
        // Defer file creation until start() when targetFormat is known
        self.pendingOutputURL = outputURL
        self.outputFile = nil
    }

    // Activate audio session, configure formats/converter, and start the engine/tap.
    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: [])

        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: inputBus)
        // Set session preferred sample rate to input/hardware
        try? AVAudioSession.sharedInstance().setPreferredSampleRate(inputFormat.sampleRate)
        targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputFormat.sampleRate, channels: 1, interleaved: false)
        converter = inputFormat.sampleRate == targetFormat.sampleRate && inputFormat.channelCount == 1 ? nil : AVAudioConverter(from: inputFormat, to: targetFormat)

        // If in record mode, open output file now that targetFormat is set
        if case .record = mode, let url = pendingOutputURL {
            self.outputFile = try AVAudioFile(forWriting: url, settings: targetFormat.settings)
            debugTotalFramesWritten = 0
            print("AudioEngine: recording to \(url.path)")
        }

        input.removeTap(onBus: inputBus)
        input.installTap(onBus: inputBus, bufferSize: 2048, format: inputFormat) { [weak self] buffer, _ in
            self?.process(buffer: buffer)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    // Stop engine and release resources. Closing the output file finalizes WAV headers.
    func stop() {
        engine.inputNode.removeTap(onBus: inputBus)
        engine.stop()
        isRunning = false
        // Release file to finalize header on deinit
        outputFile = nil
    }

    // Convert incoming buffer to target format, update RMS, and either append to
    // in-memory buffer (enroll) or stream to disk (record).
    private func process(buffer: AVAudioPCMBuffer) {
        let frameCapacity = AVAudioFrameCount(Double(buffer.frameLength) * (targetFormat?.sampleRate ?? buffer.format.sampleRate) / buffer.format.sampleRate) + 1024
        guard let targetFormat = targetFormat, let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCapacity) else { return }

        var error: NSError?
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return buffer
        }
        // Preserve converter state across buffers for smooth audio
        if let conv = converter {
            conv.convert(to: outBuffer, error: &error, withInputFrom: inputBlock)
            if let error = error { print("convert error", error); return }
        } else {
            // No conversion needed: copy channel 0 or mixdown
            outBuffer.frameLength = buffer.frameLength
            if let src = buffer.floatChannelData, let dst = outBuffer.floatChannelData {
                let frames = Int(buffer.frameLength)
                if buffer.format.channelCount == 1 {
                    for i in 0..<frames { dst[0][i] = src[0][i] }
                } else {
                    for i in 0..<frames { dst[0][i] = 0.5 * (src[0][i] + src[1][i]) }
                }
            }
        }

        guard let channel = outBuffer.floatChannelData?[0] else { return }
        let count = Int(outBuffer.frameLength)
        var newSamples: [Float] = []
        newSamples.reserveCapacity(count)
        for i in 0..<count { newSamples.append(channel[i]) }
        pendingSamples += newSamples

        // RMS for UI
        levelRMS = rms(of: newSamples)

        switch mode {
        case .enroll:
            break
        case .record:
            // Continuous write (no gating)
            if let file = outputFile { write(samples: newSamples, to: file) }
            // Keep small tail for UI
            let keep = Int(1.0 * targetFormat.sampleRate)
            if pendingSamples.count > keep { pendingSamples = Array(pendingSamples.suffix(keep)) }
        }
    }

    func copyAndResetSamples() -> [Float] {
        let s = pendingSamples
        pendingSamples.removeAll()
        return s
    }

    private func rms(of v: [Float]) -> Float {
        guard !v.isEmpty else { return 0 }
        let sum = v.reduce(Float(0)) { $0 + $1*$1 }
        return sqrt(sum / Float(v.count))
    }

    // no gating path

    // Append samples to the output WAV file, logging progress periodically.
    private func write(samples: [Float], to file: AVAudioFile) {
        guard let buf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buf.frameLength = buf.frameCapacity
        let ptr = buf.floatChannelData![0]
        for i in 0..<samples.count { ptr[i] = samples[i] }
        do {
            try file.write(from: buf)
            debugTotalFramesWritten += Int64(buf.frameLength)
            if debugTotalFramesWritten % Int64(16000 * 3) == 0 { // every ~3s
                print("AudioEngine: wrote total frames = \(debugTotalFramesWritten)")
            }
        } catch {
            print("AudioEngine write error", error)
        }
    }

    // removed absolute range helper in simplified gating

    func totalFramesWritten() -> Int64 { debugTotalFramesWritten }
}


