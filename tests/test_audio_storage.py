from stt_server.backend.component.audio_storage import SessionAudioRecorder


def test_audio_recorder_writes_async(tmp_path):
    path = tmp_path / "audio.wav"
    recorder = SessionAudioRecorder("sess", path, sample_rate=16000)
    pcm = b"\x01\x02" * 800
    recorder.append(pcm)
    recorder.append(pcm)
    saved = recorder.finalize()
    assert saved == path
    assert path.exists()
    assert recorder.bytes_written == len(pcm) * 2


def test_audio_recorder_discard_empty(tmp_path):
    path = tmp_path / "empty.wav"
    recorder = SessionAudioRecorder("sess", path, sample_rate=16000)
    saved = recorder.finalize()
    assert saved is None
    assert not path.exists()
