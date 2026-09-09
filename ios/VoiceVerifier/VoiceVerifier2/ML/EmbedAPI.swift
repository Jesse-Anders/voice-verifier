import Foundation

struct EmbedAPIResponse: Decodable {
    let embedding: [Double]
}

struct ScoreAPIResponse: Decodable {
    let score: Double
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

    func embedMean(fileURL: URL, timeout: TimeInterval = 30) throws -> [Float] {
        let endpoint = baseURL.appendingPathComponent("embed_mean")
        var request = URLRequest(url: endpoint)
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
}


