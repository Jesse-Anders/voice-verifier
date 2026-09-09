import Foundation

// Client for the FastAPI embed/score endpoints used by the app.
// Responsibilities:
// - Manage the base URL (overridable via Settings)
// - Health check for connectivity
// - Upload a WAV to /embed_mean and decode its embedding
// - Upload two WAVs to /score_cosine and decode their cosine score

struct EmbedAPIResponse: Decodable {
    let embedding: [Double]
}

struct ScoreAPIResponse: Decodable {
    let score: Double
}

struct ScoreDetailedAPIResponse: Decodable {
    let score: Double
    let duration: Double
    let threshold: Double
    let margin: Double
    let accepted: Bool
}
final class EmbedAPI {
    static let shared = EmbedAPI()
    private var baseURL: URL
    private init() {
        #if targetEnvironment(simulator)
        let defaultURL = URL(string: "http://127.0.0.1:8000")!
        #else
        let defaultURL = URL(string: "http://127.0.0.1:8000")! // override via UserDefaults on device
        #endif
        if let s = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL"), let u = URL(string: s) {
            baseURL = u
        } else {
            baseURL = defaultURL
        }
    }

    func refreshBaseURL() {
        if let s = UserDefaults.standard.string(forKey: "EmbedAPIBaseURL"), let u = URL(string: s) {
            baseURL = u
        }
    }

    // Simple GET /health -> {"ok": true}
    func health(timeout: TimeInterval = 3) -> Bool {
        let endpoint = baseURL.appendingPathComponent("health")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout

        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var ok = false
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            guard error == nil, let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else { return }
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let v = obj["ok"] as? Bool {
                ok = v
            }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        return ok
    }

    // Upload a WAV file and return the mean speaker embedding computed by the server.
    // If pp is provided, include it as a query parameter.
    func embedMean(fileURL: URL, pp: String? = nil, timeout: TimeInterval = 30) throws -> [Float] {
        let endpoint = baseURL.appendingPathComponent("embed_mean")
        var url = endpoint
        if let pp = pp, var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) {
            comps.queryItems = [URLQueryItem(name: "pp", value: pp)]
            if let u = comps.url { url = u }
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }

        let filename = fileURL.lastPathComponent
        let fileData = try Data(contentsOf: fileURL)

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"wav\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(fileData)
        append("\r\n")
        append("--\(boundary)--\r\n")

        request.httpBody = body
        request.timeoutInterval = timeout

        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[Float], Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error {
                print("EmbedAPI embedMean error: \(e.localizedDescription)")
                result = .failure(e)
                return
            }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                print("EmbedAPI embedMean http status: \(code)")
                result = .failure(NSError(domain: "EmbedAPI", code: code))
                return
            }
            do {
                let decoded = try JSONDecoder().decode(EmbedAPIResponse.self, from: data)
                result = .success(decoded.embedding.map { Float($0) })
            } catch {
                print("EmbedAPI embedMean decode error: \(error.localizedDescription)")
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

    // Upload two WAV files and return their cosine similarity from the server.
    func scoreCosine(fileA: URL, fileB: URL, timeout: TimeInterval = 30) throws -> Float {
        let endpoint = baseURL.appendingPathComponent("score_cosine")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }

        let aData = try Data(contentsOf: fileA)
        let bData = try Data(contentsOf: fileB)

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"a\"; filename=\"\(fileA.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(aData)
        append("\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"b\"; filename=\"\(fileB.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(bData)
        append("\r\n")
        append("--\(boundary)--\r\n")

        request.httpBody = body
        request.timeoutInterval = timeout

        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Float, Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error {
                print("EmbedAPI scoreCosine error: \(e.localizedDescription)")
                result = .failure(e)
                return
            }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                print("EmbedAPI scoreCosine http status: \(code)")
                result = .failure(NSError(domain: "EmbedAPI", code: code))
                return
            }
            do {
                let decoded = try JSONDecoder().decode(ScoreAPIResponse.self, from: data)
                result = .success(Float(decoded.score))
            } catch {
                print("EmbedAPI scoreCosine decode error: \(error.localizedDescription)")
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

    // Single-shot embedding
    func embedSingle(fileURL: URL, pp: String? = nil, timeout: TimeInterval = 60) throws -> [Float] {
        let endpoint = baseURL.appendingPathComponent("embed_single")
        var url = endpoint
        if let pp = pp, var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) {
            comps.queryItems = [URLQueryItem(name: "pp", value: pp)]
            if let u = comps.url { url = u }
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let filename = fileURL.lastPathComponent
        let fileData = try Data(contentsOf: fileURL)
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"wav\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(fileData)
        append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[Float], Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                result = .failure(NSError(domain: "EmbedAPI", code: code)); return
            }
            do {
                let decoded = try JSONDecoder().decode(EmbedAPIResponse.self, from: data)
                result = .success(decoded.embedding.map { Float($0) })
            } catch { result = .failure(error) }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }

    // Single-shot cosine for two WAVs on the server (no slicing)
    func scoreCosineSingle(trainURL: URL, testURL: URL, pp: String? = nil, duration: Double? = nil, log: Int = 0, timeout: TimeInterval = 60) throws -> ScoreDetailedAPIResponse {
        let endpoint = baseURL.appendingPathComponent("score_cosine_single")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        var q: [URLQueryItem] = []
        if let pp = pp { q.append(URLQueryItem(name: "pp", value: pp)) }
        if let d = duration { q.append(URLQueryItem(name: "dur_sec", value: String(format: "%.6f", d))) }
        q.append(URLQueryItem(name: "log", value: "\(log)"))
        comps.queryItems = q.isEmpty ? nil : q
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let aData = try Data(contentsOf: trainURL)
        let bData = try Data(contentsOf: testURL)
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"a\"; filename=\"\(trainURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(aData); append("\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"b\"; filename=\"\(testURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(bData); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<ScoreDetailedAPIResponse, Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                result = .failure(NSError(domain: "EmbedAPI", code: code)); return
            }
            do {
                let decoded = try JSONDecoder().decode(ScoreDetailedAPIResponse.self, from: data)
                result = .success(decoded)
            } catch { result = .failure(error) }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }

    // Slice-based single-shot scoring handled server-side; avoids client slicing
    func scoreCosineSlice(trainURL: URL, testURL: URL, start: Double, end: Double, pp: String? = nil, log: Int = 0, timeout: TimeInterval = 60) throws -> ScoreDetailedAPIResponse {
        let endpoint = baseURL.appendingPathComponent("score_cosine_slice")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        var q: [URLQueryItem] = [URLQueryItem(name: "s", value: String(format: "%.6f", start)),
                                 URLQueryItem(name: "e", value: String(format: "%.6f", end)),
                                 URLQueryItem(name: "log", value: "\(log)")]
        if let pp = pp { q.append(URLQueryItem(name: "pp", value: pp)) }
        comps.queryItems = q
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let aData = try Data(contentsOf: trainURL)
        let bData = try Data(contentsOf: testURL)
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"train\"; filename=\"\(trainURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(aData); append("\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"test\"; filename=\"\(testURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(bData); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<ScoreDetailedAPIResponse, Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                result = .failure(NSError(domain: "EmbedAPI", code: code)); return
            }
            do {
                let decoded = try JSONDecoder().decode(ScoreDetailedAPIResponse.self, from: data)
                result = .success(decoded)
            } catch { result = .failure(error) }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }

    // Batch version: one upload, many [start,end] slices scored server-side
    func scoreCosineSliceBatch(trainURL: URL, testURL: URL, slices: [[Double]], pp: String? = nil, log: Int = 0, timeout: TimeInterval = 90) throws -> [ScoreDetailedAPIResponse] {
        let endpoint = baseURL.appendingPathComponent("score_cosine_slice_batch")
        var comps = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        var q: [URLQueryItem] = [URLQueryItem(name: "log", value: "\(log)")]
        if let pp = pp { q.append(URLQueryItem(name: "pp", value: pp)) }
        comps.queryItems = q
        var request = URLRequest(url: comps.url!)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func append(_ s: String) { body.append(s.data(using: .utf8)!) }
        let aData = try Data(contentsOf: trainURL)
        let bData = try Data(contentsOf: testURL)
        let slicesJSON = try String(data: JSONSerialization.data(withJSONObject: slices), encoding: .utf8) ?? "[]"
        // train
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"train\"; filename=\"\(trainURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(aData); append("\r\n")
        // test
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"test\"; filename=\"\(testURL.lastPathComponent)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(bData); append("\r\n")
        // slices JSON (as regular form field)
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"slices\"\r\n\r\n")
        append(slicesJSON); append("\r\n")
        append("--\(boundary)--\r\n")
        request.httpBody = body
        request.timeoutInterval = timeout
        let session = URLSession(configuration: .default)
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<[ScoreDetailedAPIResponse], Error> = .failure(NSError(domain: "EmbedAPI", code: -1))
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            if let e = error { result = .failure(e); return }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode), let data = data else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? -2
                result = .failure(NSError(domain: "EmbedAPI", code: code)); return
            }
            do {
                let decoded = try JSONDecoder().decode([ScoreDetailedAPIResponse].self, from: data)
                result = .success(decoded)
            } catch { result = .failure(error) }
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout)
        switch result {
        case .success(let v): return v
        case .failure(let e): throw e
        }
    }
}


