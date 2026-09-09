import Foundation

struct Memo: Codable, Identifiable, Equatable {
    let id: UUID
    let profileId: UUID
    let createdAt: Date
    var title: String?
    let originalFile: String      // relative to Documents
    let concatFile: String        // relative to Documents
    let plotFile: String          // relative to Documents
    let summaryFile: String?      // relative to Documents
    let mode: String              // "amp" or "scd"
    let pp: String                // e.g., "none"
}

final class MemoStore: ObservableObject {
    static let shared = MemoStore()
    private init() {}

    private func baseDir() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("Memos")
        if !FileManager.default.fileExists(atPath: dir.path) { try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
        return dir
    }

    private func listURL(for profileId: UUID) -> URL {
        baseDir().appendingPathComponent("\(profileId.uuidString).json")
    }

    func load(profileId: UUID) -> [Memo] {
        let url = listURL(for: profileId)
        guard let data = try? Data(contentsOf: url), let arr = try? JSONDecoder().decode([Memo].self, from: data) else { return [] }
        return arr.sorted { $0.createdAt > $1.createdAt }
    }

    func add(profileId: UUID, original: URL, concat: URL, plot: URL, summary: URL?, mode: String, pp: String) {
        // Ensure files are stored under Documents/Memos/<profileId>/<memoId>/
        let memoId = UUID()
        let root = baseDir().appendingPathComponent(profileId.uuidString)
        if !FileManager.default.fileExists(atPath: root.path) { try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true) }
        let memoDir = root.appendingPathComponent(memoId.uuidString)
        try? FileManager.default.createDirectory(at: memoDir, withIntermediateDirectories: true)

        func copy(_ src: URL, name: String) -> URL {
            let dst = memoDir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dst)
            try? FileManager.default.copyItem(at: src, to: dst)
            return dst
        }

        let origDst = copy(original, name: original.lastPathComponent)
        let concatDst = copy(concat, name: concat.lastPathComponent)
        let plotDst = copy(plot, name: plot.lastPathComponent)
        var summaryRel: String? = nil
        if let s = summary, FileManager.default.fileExists(atPath: s.path) {
            let sdst = copy(s, name: s.lastPathComponent)
            summaryRel = relativePath(of: sdst)
        }

        let memo = Memo(
            id: memoId,
            profileId: profileId,
            createdAt: Date(),
            title: nil,
            originalFile: relativePath(of: origDst),
            concatFile: relativePath(of: concatDst),
            plotFile: relativePath(of: plotDst),
            summaryFile: summaryRel,
            mode: mode,
            pp: pp
        )
        var list = readList(profileId: profileId)
        list.append(memo)
        save(list: list, for: profileId)
    }

    private func save(list: [Memo], for profileId: UUID) {
        let url = listURL(for: profileId)
        if let data = try? JSONEncoder().encode(list) { try? data.write(to: url, options: .atomic) }
    }

    private func readList(profileId: UUID) -> [Memo] {
        let url = listURL(for: profileId)
        guard let data = try? Data(contentsOf: url), let arr = try? JSONDecoder().decode([Memo].self, from: data) else { return [] }
        return arr
    }

    private func relativePath(of url: URL) -> String {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let p = url.path
        if p.hasPrefix(docs.path) {
            return String(p.dropFirst(docs.path.count + (docs.path.hasSuffix("/") ? 0 : 1)))
        }
        return url.lastPathComponent
    }

    func absoluteURL(forRelative rel: String) -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(rel)
    }

    func delete(profileId: UUID, memo: Memo) {
        // Remove memo directory if possible
        let anyPath = memo.plotFile
        let dir = absoluteURL(forRelative: anyPath).deletingLastPathComponent()
        try? FileManager.default.removeItem(at: dir)
        // Update index
        var list = readList(profileId: profileId)
        list.removeAll { $0.id == memo.id }
        save(list: list, for: profileId)
    }

    func rename(profileId: UUID, memoId: UUID, title: String) {
        var list = readList(profileId: profileId)
        if let idx = list.firstIndex(where: { $0.id == memoId }) {
            list[idx].title = title
            save(list: list, for: profileId)
        }
    }
}


