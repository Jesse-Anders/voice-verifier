import SwiftUI
import AVFoundation

struct MemoDetailView: View {
    let memo: Memo

    @State private var player: AVAudioPlayer?
    @State private var showRename = false
    @State private var newTitle: String = ""
    @Environment(\.dismiss) private var dismiss
    @State private var showShare = false
    @State private var shareURL: URL? = nil
    @State private var showDeleteConfirm = false

    private func urlFor(_ rel: String) -> URL {
        MemoStore.shared.absoluteURL(forRelative: rel)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Show memo title/date
                Text((memo.title?.isEmpty == false ? memo.title! + " • " : "") + memo.createdAt.formatted())
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                // Show original filename prominently at top-left
                Text(urlFor(memo.originalFile).lastPathComponent)
                    .font(.headline)
                    .foregroundStyle(.primary)

                // Plot PNG
                if let img = UIImage(contentsOfFile: urlFor(memo.plotFile).path) {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(6)
                    HStack {
                        Spacer()
                        Button("Share Visual") {
                            shareURL = urlFor(memo.plotFile)
                            showShare = true
                        }
                    }
                }

                GroupBox("Original") {
                    HStack {
                        Text(urlFor(memo.originalFile).lastPathComponent).lineLimit(1)
                        Spacer()
                        Button(player?.isPlaying == true ? "Stop" : "Play") { togglePlay(urlFor(memo.originalFile)) }
                        Button("Share") { shareURL = urlFor(memo.originalFile); showShare = true }
                    }
                }

                GroupBox("Concatenated") {
                    HStack {
                        Text(urlFor(memo.concatFile).lastPathComponent).lineLimit(1)
                        Spacer()
                        Button(player?.isPlaying == true ? "Stop" : "Play") { togglePlay(urlFor(memo.concatFile)) }
                        Button("Share") { shareURL = urlFor(memo.concatFile); showShare = true }
                    }
                }

                if let s = memo.summaryFile {
                    GroupBox("Summary JSON") {
                        Text(urlFor(s).lastPathComponent)
                        Button("Share") { shareURL = urlFor(s); showShare = true }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Memo")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button {
                    newTitle = memo.title ?? ""
                    showRename = true
                } label: { Image(systemName: "pencil") }
                Button {
                    showDeleteConfirm = true
                } label: { Image(systemName: "trash") }
            }
        }
        .alert("Delete this memo?", isPresented: $showDeleteConfirm) {
            Button("Delete", role: .destructive) {
                MemoStore.shared.delete(profileId: memo.profileId, memo: memo)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will remove the memo and its files from your device.")
        }
        .sheet(isPresented: $showRename) {
            VStack(spacing: 16) {
                Text("Rename Memo").font(.headline)
                TextField("Title", text: $newTitle).textFieldStyle(.roundedBorder)
                HStack {
                    Button("Cancel") { showRename = false }
                    Spacer()
                    Button("Save") {
                        MemoStore.shared.rename(profileId: memo.profileId, memoId: memo.id, title: newTitle)
                        showRename = false
                    }
                }
            }
            .padding()
            .presentationDetents([.height(180)])
        }
        .sheet(isPresented: $showShare) {
            if let u = shareURL {
                ActivityView(activityItems: [u])
            }
        }
    }

    private func togglePlay(_ url: URL) {
        if let p = player, p.isPlaying {
            p.stop(); player = nil
        } else {
            do { let p = try AVAudioPlayer(contentsOf: url); p.prepareToPlay(); p.play(); player = p } catch { print("player error", error) }
        }
    }

}


