import Foundation

final class PlotAPI {
    static let shared = PlotAPI()
    private init() {}

    private func baseURL() -> URL {
        if let s = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL"), let u = URL(string: s) {
            return u
        }
        return URL(string: "http://127.0.0.1:8000")!
    }

    // Standard plot (Amplitude only)
    func plotPair(train: URL, test: URL, pp: String = "none", timeout: TimeInterval = 60) throws -> Data {
        let endpoint = baseURL().appendingPathComponent("plot_pair")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        comps.queryItems = [
            URLQueryItem(name: "pp", value: pp),
            URLQueryItem(name: "smooth", value: "1")
        ]
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let trainData = try Data(contentsOf: train)
        let testData = try Data(contentsOf: test)
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"train\"; filename=\"\(train.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(trainData); append("\r\n")
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"test\"; filename=\"\(test.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(testData); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Data, Error> = .failure(NSError(domain: "PlotAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                result = .failure(NSError(domain: "PlotAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -2))
                return
            }
            result = .success(data)
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let d): return d
        case .failure(let e): throw e
        }
    }

    // Sliding plot (VAD + window/hop)
    // If winSec/hopSec/pp are nil, server uses its env defaults
    func plotPairSliding(train: URL, test: URL, winSec: Double? = nil, hopSec: Double? = nil, pp: String? = nil, timeout: TimeInterval = 60) throws -> Data {
        let endpoint = baseURL().appendingPathComponent("plot_pair_sliding")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        var items: [URLQueryItem] = []
        if let w = winSec { items.append(URLQueryItem(name: "win_sec", value: String(w))) }
        if let h = hopSec { items.append(URLQueryItem(name: "hop_sec", value: String(h))) }
        if let pp = pp { items.append(URLQueryItem(name: "pp", value: pp)) }
        comps.queryItems = items.isEmpty ? nil : items
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let trainData = try Data(contentsOf: train)
        let testData = try Data(contentsOf: test)
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"train\"; filename=\"\(train.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(trainData); append("\r\n")
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"test\"; filename=\"\(test.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(testData); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Data, Error> = .failure(NSError(domain: "PlotAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                result = .failure(NSError(domain: "PlotAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -2))
                return
            }
            result = .success(data)
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let d): return d
        case .failure(let e): throw e
        }
    }

    // Render plot from app's saved decisions JSON (no ML server-side)
    func plotFromDecisions(test: URL, decisionsJSON: Data, metaJSON: Data? = nil, timeout: TimeInterval = 60) throws -> Data {
        let endpoint = baseURL().appendingPathComponent("plot_from_decisions")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }

        let testData = try Data(contentsOf: test)
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"test\"; filename=\"\(test.lastPathComponent)\"\r\n"); append("Content-Type: audio/wav\r\n\r\n"); body.append(testData); append("\r\n")
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"decisions\"\r\n\r\n"); body.append(decisionsJSON); append("\r\n")
        if let metaJSON = metaJSON {
            append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"meta\"\r\n\r\n"); body.append(metaJSON); append("\r\n")
        }
        append("--\(boundary)--\r\n")

        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Data, Error> = .failure(NSError(domain: "PlotAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                result = .failure(NSError(domain: "PlotAPI", code: (response as? HTTPURLResponse)?.statusCode ?? -2))
                return
            }
            result = .success(data)
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let d): return d
        case .failure(let e): throw e
        }
    }
}
// (Removed duplicate PlotAPI declaration)


