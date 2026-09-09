# Scrub check (pre-publish gate)

Reviewed before the first push. This repo was assembled by copying only source,
config, and the staged README/images out of a 2.5 GB working directory — never
the working tree wholesale. Confirmed the published repo contains **none** of the
following:

## Excluded — verified absent from the committed tree

| Category | What was withheld |
|---|---|
| **Real audio** | 74 voice recordings (`.wav/.m4a/.caf/…`) — enrollment/test clips of real people (profiles named Jenn, Dave, Jesse, etc.). None ship. The staged screenshots/chart show the UI and aggregate scores only. |
| **Model weights** | 16 `.pt` files (SpeechBrain ECAPA + scorer/TorchScript exports, ~300 MB). Not committed — the server downloads ECAPA from SpeechBrain on first run, and the on-device TorchScript model is regenerated with `server/export_torchscript.py`. |
| **Secrets / network** | Hardcoded private LAN IPs (`192.168.x.x`) lived only in `.rtf` dev notes and one plotting script — those notes are excluded and the server is configured at runtime via **Settings → Server Base URL**. No `.env`, keys, signing certs, or provisioning profiles. |
| **Personal / raw data** | `data/`, `OLD Models PT files/`, notebook checkpoints, JSON decision summaries tied to real recordings — all excluded. |
| **Repo cruft** | Nested `.git` repos, vendored `Pods/`, `build.log`, `DerivedData`, `*.xcuserstate`, an unreferenced backup app (`VoiceVerifier-before-all-serverside/`), and a stray `RecordingView8-29-bu.swift`. |

## `.gitignore` enforces going forward
Covers audio (`*.wav *.m4a *.caf *.mp4 *.mov`), weights (`*.pt *.pth *.ckpt`,
`pretrained_models/`), secrets (`.env`, certs/keys), `Pods/`, Xcode build output,
`__pycache__/`, and `.DS_Store`.

## What ships
- `README.md`, `images/` (3 PNGs — UI + the duration-aware-threshold chart), `LICENSE`.
- `server/` — `embed_server.py`, `export_torchscript.py`, pinned `requirements.txt`.
- `ios/` — the Swift/SwiftUI Xcode workspace (both `VoiceVerifier` and
  `VoiceVerifier2` targets), source only, model/audio resources stripped.

## Scan
```
audio/weights/secrets/.rtf in tree ...... none
nested .git / .DS_Store .................. none
LAN IPs / non-local URLs in shipped code  none
```
