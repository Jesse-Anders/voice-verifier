import Foundation

struct SegmentsResponse: Decodable {
    let segments: [[Double]]
}

struct SlidingConfig: Decodable {
    let win_sec: Double
    let hop_sec: Double
    let min_keep_sec: Double
    let top_db: Double
    let pp: String
    let end_hop_offset: Int
    let tail_trim_sec: Double
    let head_trim_sec: Double
    // Optional tail drill params (present on newer servers)
    let tail_drill_step_sec: Double?
    let tail_drill_min_len_sec: Double?
    let tail_drill_max_sec: Double?
    let tail_drill_eps: Double?
    let tail_drill_vwin_sec: Double?
}

// Client for server-side segmentation endpoints.
// - Supports two modes: amplitude (/segments) and sliding (/segments_sliding)
// - Mode is stored in UserDefaults so the toggle persists across launches
final class SegmentsAPI {
    static let shared = SegmentsAPI()
    private init() {}

    static let modeKey = "SegmentationMode"

    enum SegmentationMode: String, CaseIterable, Identifiable {
        case amplitude = "amp"
        case sliding = "sliding"
        var id: String { rawValue }
        var title: String {
            switch self {
            case .amplitude: return "Amplitude"
            case .sliding: return "Sliding (VAD+Win/Hop)"
            }
        }
    }

    private func baseURL() -> URL {
        if let s = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL"), let u = URL(string: s) {
            return u
        }
        return URL(string: "http://127.0.0.1:8000")!
    }

    func slidingConfig(timeout: TimeInterval = 5) -> SlidingConfig? {
        let endpoint = baseURL().appendingPathComponent("sliding_config")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var value: SlidingConfig? = nil
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            guard error == nil,
                  let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                  let data = data,
                  let cfg = try? JSONDecoder().decode(SlidingConfig.self, from: data) else { return }
            value = cfg
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        return value
    }

    func currentMode() -> SegmentationMode {
        if let s = UserDefaults.standard.string(forKey: SegmentsAPI.modeKey) {
            if s == SegmentationMode.sliding.rawValue { return .sliding }
            return .amplitude
        }
        return .amplitude
    }

    func setMode(_ mode: SegmentationMode) {
        UserDefaults.standard.set(mode.rawValue, forKey: SegmentsAPI.modeKey)
    }

    // Sliding segmentation: send both train and test files
    // If winSec/hopSec are nil, server env defaults are used
    func slidingSegments(trainURL: URL, testURL: URL, winSec: Double? = nil, hopSec: Double? = nil, timeout: TimeInterval = 120) throws -> [[Double]] {
        let endpoint = baseURL().appendingPathComponent("segments_sliding")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        var items: [URLQueryItem] = []
        if let w = winSec { items.append(URLQueryItem(name: "win_sec", value: String(w))) }
        if let h = hopSec { items.append(URLQueryItem(name: "hop_sec", value: String(h))) }
        comps.queryItems = items.isEmpty ? nil : items
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let trainData = try Data(contentsOf: trainURL)
        let testData = try Data(contentsOf: testURL)
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"train\"; filename=\"\(trainURL.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(trainData); append("\r\n")
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"test\"; filename=\"\(testURL.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(testData); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[[Double]], Error> = .failure(NSError(domain: "SegmentsAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                result = .failure(NSError(domain: "SegmentsAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -2))
                return
            }
            do {
                let decoded = try JSONDecoder().decode(SegmentsResponse.self, from: data)
                result = .success(decoded.segments)
            } catch {
                result = .failure(error)
            }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }


    // Post a WAV to the selected endpoint and return [[start,end]] seconds.
    func segments(fileURL: URL, mode: SegmentationMode? = nil, timeout: TimeInterval = 0) throws -> [[Double]] {
        let m = mode ?? currentMode()
        let endpoint = baseURL().appendingPathComponent(m == .sliding ? "segments_sliding" : "segments")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }

        let filename = fileURL.lastPathComponent
        let fileData = try Data(contentsOf: fileURL)

        if m == .sliding {
            // Sliding requires train+test. For this generic method we send 'test' only; caller should prefer slidingSegments(...)
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"test\"; filename=\"\(filename)\"\r\n")
            append("Content-Type: audio/wav\r\n\r\n")
            body.append(fileData)
            append("\r\n")
        } else {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"wav\"; filename=\"\(filename)\"\r\n")
            append("Content-Type: audio/wav\r\n\r\n")
            body.append(fileData)
            append("\r\n")
        }
        append("--\(boundary)--\r\n")

        request.httpBody = body
        let effectiveTimeout: TimeInterval = (timeout > 0) ? timeout : (m == .sliding ? 120 : 45)
        request.timeoutInterval = effectiveTimeout

        print("SegmentsAPI: mode=\(m.rawValue) url=\(endpoint.absoluteString) timeout=\(Int(effectiveTimeout))s file=\(fileURL.lastPathComponent)")
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[[Double]], Error> = .failure(NSError(domain: "SegmentsAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                result = .failure(NSError(domain: "SegmentsAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -2))
                return
            }
            do {
                let decoded = try JSONDecoder().decode(SegmentsResponse.self, from: data)
                result = .success(decoded.segments)
            } catch {
                result = .failure(error)
            }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + effectiveTimeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }
}


