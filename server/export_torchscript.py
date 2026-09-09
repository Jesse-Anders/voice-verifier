import os
import torch
import torchaudio

# pip install speechbrain soundfile librosa torch
from speechbrain.inference.speaker import EncoderClassifier
import argparse


def export_ecapa_embedding_torchscript(out_path: str, sample_len_sec: float = 3.0, sr: int = 16000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load SpeechBrain ECAPA encoder
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    class EncodeWrapper(torch.nn.Module):
        def __init__(self, enc: EncoderClassifier):
            super().__init__()
            self.enc = enc

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            return self.enc.encode_batch(wav)

    model = EncodeWrapper(classifier)
    model.eval()

    # Trace the SpeechBrain encoder wrapper (SB is not fully TorchScript-scriptable)
    T = int(sample_len_sec * sr)
    example = torch.zeros(T, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)
        traced.save(out_path)
    print(f"Saved traced TorchScript model to {out_path}")


def export_scorer(out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    class Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
            self.win = int(3.0 * 16000)
            self.hop = int(1.0 * 16000)

        def trim(self, x: torch.Tensor) -> torch.Tensor:
            sr = 16000
            frame = max(1, int(0.03 * sr))
            hop = max(1, int(0.01 * sr))
            T = x.shape[0]
            if T < frame:
                return x
            w = torch.ones(1, 1, frame, dtype=x.dtype)
            xsq = x.view(1, 1, -1).pow(2)
            eng = torch.nn.functional.conv1d(xsq, w, stride=hop).squeeze()
            rms = torch.sqrt(eng / frame + 1e-12)
            db = 20.0 * torch.log10(torch.clamp(rms, min=1e-6))
            max_db = torch.max(db)
            thr = max_db - 25.0
            idx = torch.nonzero(db > thr).flatten()
            if idx.numel() == 0:
                return x
            start = int(idx[0].item()) * hop
            end = min(T, int(idx[-1].item()) * hop + frame)
            return x[start:end]

        def forward(self, wav: torch.Tensor, input_sr: torch.Tensor) -> torch.Tensor:
            if wav.dim() != 1:
                wav = wav.squeeze()
            sr = int(input_sr.item())
            if sr != 16000:
                resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav16 = resample(wav.unsqueeze(0)).squeeze(0)
            else:
                wav16 = wav

            x = self.trim(wav16)
            T = x.shape[0]
            if T < self.win:
                out = torch.zeros(self.win, dtype=x.dtype)
                out[:T] = x
                chunks = out.unsqueeze(0)
            else:
                starts = torch.arange(0, T - self.win + 1, self.hop)
                chunks = torch.stack([x[s:s + self.win] for s in starts], dim=0)

            embs = []
            with torch.no_grad():
                for i in range(chunks.shape[0]):
                    e = self.enc.encode_batch(chunks[i].unsqueeze(0)).squeeze(0)
                    e = e / (torch.linalg.norm(e) + 1e-12)
                    embs.append(e)
            E = torch.stack(embs, dim=0)
            mean = torch.mean(E, dim=0)
            mean = mean / (torch.linalg.norm(mean) + 1e-12)
            return mean

    scorer = Scorer().eval()
    example_wav = torch.zeros(48000, dtype=torch.float32)
    example_sr = torch.tensor(48000, dtype=torch.int64)
    with torch.no_grad():
        ts = torch.jit.trace(scorer, (example_wav, example_sr), strict=False)
        ts.save(out_path)
    print(f"Saved traced TorchScript scorer to {out_path}")


# -----------------------------
# Scriptable ECAPA (Torch-only frontend) -> embedding_model
# Avoids SpeechBrain feature graph and torchaudio ops; produces a fully torch.jit.script-able module
def export_ecapa_embedding_scriptable(out_path: str, sr: int = 16000, n_mels: int = 80, n_fft: int = 512, win_ms: float = 25.0, hop_ms: float = 10.0):
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    enc = classifier.mods.embedding_model.eval()  # torch.nn.Module (ECAPA_TDNN)
    enc.to(device)

    win_length = int(sr * win_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)
    window = torch.hann_window(win_length).to(device)

    # Build Mel filterbank in torch (Slaney-style)
    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def build_mel(sr, n_fft, n_mels, fmin=0.0, fmax=None):
        if fmax is None:
            fmax = sr / 2.0
        m_min = hz_to_mel(torch.tensor([fmin]))[0]
        m_max = hz_to_mel(torch.tensor([fmax]))[0]
        m_pts = torch.linspace(m_min, m_max, n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        bins = torch.floor((n_fft // 2 + 1) * f_pts / (sr / 2.0)).long()
        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center == left: center += 1
            if right == center: right += 1
            fb[i, left:center] = torch.linspace(0, 1, max(1, center - left))
            fb[i, center:right] = torch.linspace(1, 0, max(1, right - center))
        # Slaney mel area normalization
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb = fb * enorm.unsqueeze(1)
        return fb

    mel_fb = build_mel(sr, n_fft, n_mels).to(device)

    class ScriptableECAPAWrapper(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module, sr: int, n_fft: int, win_length: int, hop_length: int, window: torch.Tensor, mel_fb: torch.Tensor):
            super().__init__()
            self.encoder = encoder
            self.sr = sr
            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.register_buffer("window", window)
            self.register_buffer("mel_fb", mel_fb)

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            # wav: 1D float tensor at 16 kHz
            if wav.dim() != 1:
                wav = wav.view(-1)
            # STFT -> power spec
            stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=True, return_complex=True)
            mag = (stft.real.pow(2) + stft.imag.pow(2))  # [freq, frames]
            # Mel projection
            mel_spec = torch.matmul(self.mel_fb, mag)  # [n_mels, frames]
            mel_spec = torch.clamp(mel_spec, min=1e-10).log()
            # CMVN per utterance (time axis = 1)
            mean = mel_spec.mean(dim=1, keepdim=True)
            std = mel_spec.std(dim=1, unbiased=False, keepdim=True) + 1e-5
            mel_norm = (mel_spec - mean) / std
            # ECAPA expects (batch, time, feat) or (batch, feat, time) depending on implementation
            # SpeechBrain ECAPA_TDNN expects (batch, time, features)
            feats = mel_norm.transpose(0, 1).unsqueeze(0)  # [1, frames, n_mels]
            emb = self.encoder(feats)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            return emb.squeeze(0)

    wrapper = ScriptableECAPAWrapper(enc, sr, n_fft, win_length, hop_length, window, mel_fb).eval()
    wrapper.to(device)

    # Trace the torch-only frontend + SB encoder (scripting fails due to try/except in encoder)
    example_wav = torch.randn(int(3.0 * sr), dtype=torch.float32).to(device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_wav, strict=False)
        traced.save(out_path)
    print(f"Saved TRACED ECAPA embedding (torch-only frontend) to {out_path}")


def export_ecapa_embedding_kaldi_fbank(out_path: str, sr: int = 16000):
    """Trace a wrapper that uses torchaudio.compliance.kaldi.fbank + SB ECAPA encoder.
    This mirrors SpeechBrain's typical frontend more closely than plain STFT+mel.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    enc = classifier.mods.embedding_model.eval().to(device)

    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device) if sr != 16000 else None

    class KaldiFBankWrapper(torch.nn.Module):
        def __init__(self, encoder, resample):
            super().__init__()
            self.encoder = encoder
            self.resample = resample

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            # wav: 1D float tensor at sr (possibly != 16k)
            if wav.dim() != 1:
                wav = wav.view(-1)
            x = wav.unsqueeze(0)  # [1, T]
            if self.resample is not None:
                x = self.resample(x)
            x = x.squeeze(0)
            # Kaldi fbank -> [frames, mel]
            fb = torchaudio.compliance.kaldi.fbank(
                waveform=x.unsqueeze(0),
                sample_frequency=16000,
                frame_length=25.0,
                frame_shift=10.0,
                num_mel_bins=80,
                use_energy=False,
                window_type='hamming',
                dither=0.0,
                snip_edges=True,
                use_log_fbank=True,
            )
            # CMVN per utterance along time axis (dim=0 is frames)
            mean = fb.mean(dim=0, keepdim=True)
            std = fb.std(dim=0, unbiased=False, keepdim=True) + 1e-5
            fb = (fb - mean) / std
            feats = fb.unsqueeze(0)  # [1, frames, mel]
            emb = self.encoder(feats)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            return emb.squeeze(0)

    wrapper = KaldiFBankWrapper(enc, resample).eval().to(device)

    with torch.no_grad():
        example = torch.randn(int(3.0 * sr), dtype=torch.float32).to(device)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced.save(out_path)
    print(f"Saved TRACED ECAPA embedding (kaldi fbank frontend) to {out_path}")


def export_ecapa_from_sb_modules(out_path: str, sr: int = 16000):
    """Trace a wrapper that stitches SB's own compute_features + mean_var_norm + embedding_model.
    This mirrors SpeechBrain exactly while remaining trace-able.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    feats_mod = classifier.mods.compute_features.eval().to(device)
    cmvn_mod = classifier.mods.mean_var_norm.eval().to(device)
    enc = classifier.mods.embedding_model.eval().to(device)
    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device) if sr != 16000 else None

    class SBWrapper(torch.nn.Module):
        def __init__(self, feats, cmvn, encoder, resample):
            super().__init__()
            self.feats = feats
            self.cmvn = cmvn
            self.encoder = encoder
            self.resample = resample

        def forward(self, wav: torch.Tensor) -> torch.Tensor:
            # wav: 1D float tensor at sr
            if wav.dim() != 1:
                wav = wav.view(-1)
            x = wav.unsqueeze(0)  # [1, T]
            if self.resample is not None:
                x = self.resample(x)
            # SB features: [B, frames, mel]
            feats = self.feats(x)
            # CMVN requires lengths; pass full-length (1.0)
            lengths = torch.tensor([1.0], dtype=torch.float32, device=feats.device)
            feats = self.cmvn(feats, lengths)
            emb = self.encoder(feats)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            return emb.squeeze(0)

    wrapper = SBWrapper(feats_mod, cmvn_mod, enc, resample).eval().to(device)

    with torch.no_grad():
        example = torch.randn(int(3.0 * sr), dtype=torch.float32).to(device)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced.save(out_path)
    print(f"Saved TRACED ECAPA embedding (SB modules pipeline) to {out_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "Model")
    os.makedirs(model_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Export ECAPA TorchScript models")
    parser.add_argument("--mode", choices=[
        "embed_trace",     # wrapper.encode_batch traced (original)
        "scorer",          # end-to-end scorer (traced)
        "scriptable",      # torch-only STFT+mel frontend + ECAPA (traced)
        "kaldi",           # kaldi fbank frontend + ECAPA (traced)
        "sbtrace"          # SB compute_features + cmvn + ECAPA (traced)
    ], default="embed_trace")
    parser.add_argument("--out", default=None, help="Output .pt path (defaults to Model/<name>.pt)")
    parser.add_argument("--sr", type=int, default=16000, help="Input sample rate for exporters that resample")
    args = parser.parse_args()

    if args.mode == "embed_trace":
        out = args.out or os.path.join(model_dir, "ecapa_embedding.pt")
        export_ecapa_embedding_torchscript(out)
    elif args.mode == "scorer":
        out = args.out or os.path.join(model_dir, "scorer.pt")
        export_scorer(out)
    elif args.mode == "scriptable":
        out = args.out or os.path.join(model_dir, "ecapa_embedding_scripted.pt")
        export_ecapa_embedding_scriptable(out, sr=args.sr)
    elif args.mode == "kaldi":
        out = args.out or os.path.join(model_dir, "ecapa_embedding_kaldi.pt")
        export_ecapa_embedding_kaldi_fbank(out, sr=args.sr)
    elif args.mode == "sbtrace":
        out = args.out or os.path.join(model_dir, "ecapa_embedding_sbtrace.pt")
        export_ecapa_from_sb_modules(out, sr=args.sr)


