# Model files are not committed

TorchScript weights are intentionally excluded from this repo (see `SCRUB_CHECK.md`).

The default flow embeds audio via the FastAPI server, which downloads the
SpeechBrain ECAPA model on first run — no bundled model needed.

For the optional on-device verifier (`SpeakerVerifier`, bypassed by default),
generate the TorchScript model and add it to this group in Xcode:

```bash
python server/export_torchscript.py --out ecapa_embedding.pt
```
