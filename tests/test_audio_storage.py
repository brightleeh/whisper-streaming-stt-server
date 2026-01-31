from stt_server.backend.component.audio_storage import SessionAudioRecorder


def test_audio_recorder_writes_async(tmp_path):
    """Test audio recorder writes async."""
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
    """Test audio recorder discard empty."""
    path = tmp_path / "empty.wav"
    recorder = SessionAudioRecorder("sess", path, sample_rate=16000)
    saved = recorder.finalize()
    assert saved is None
    assert not path.exists()


def test_audio_recorder_queue_maxsize(tmp_path):
    """Test audio recorder queue maxsize."""
    path = tmp_path / "queue.wav"
    recorder = SessionAudioRecorder("sess", path, sample_rate=16000, queue_max_chunks=1)
    assert recorder._queue.maxsize == 1
    recorder.finalize()
