"""Audio/video transcription for ingestion (extra: `audio` -> faster-whisper)."""
from __future__ import annotations


class WhisperTranscriber:
    """Callable `path -> [(start_s, end_s, text)]` for Memory.ingest(transcriber=...)."""

    def __init__(self, model: str = "base", device: str = "cpu", compute_type: str = "int8",
                 language: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError("transcription needs the 'audio' extra: pip install 'playmemory[audio]'") from e
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language

    def __call__(self, path: str) -> list[tuple[float, float, str]]:
        segments, _info = self._model.transcribe(path, language=self.language, vad_filter=True)
        return [(s.start, s.end, s.text) for s in segments]
