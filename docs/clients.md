# Client SDKs & UI Examples

This repo ships a minimal Python SDK.

## Python SDK (streaming + retry + keepalive)

Location: `stt_client/sdk/`

```python
from stt_client.sdk import StreamingClient, RetryConfig
from gen.stt.python.v1 import stt_pb2

client = StreamingClient(
    "localhost:50051",
    keepalive_time_ms=30000,
    keepalive_timeout_ms=10000,
    signed_token_secret="your-secret",
)

session_id = "demo-session"
metadata = client.build_signed_metadata(session_id)

request = stt_pb2.SessionRequest(
    session_id=session_id,
    vad_mode=stt_pb2.VAD_CONTINUE,
    decode_profile=stt_pb2.DECODE_PROFILE_REALTIME,
    attributes={"partial": "true", "emit_final_on_vad": "true"},
)
client.create_session(request, metadata=metadata, retry=RetryConfig(attempts=3))

def audio_iter():
    # yield stt_pb2.AudioChunk(...) items
    yield stt_pb2.AudioChunk(
        session_id=session_id,
        sample_rate=16000,
        pcm16=b"...",
        is_final=True,
    )

for result in client.streaming_recognize(audio_iter(), metadata=metadata):
    print(result.committed_text, result.unstable_text)
```

Set `attributes.emit_final_on_vad=true` to emit **final** results on every VAD trigger
without ending the stream (useful for multi-utterance sessions).

For offline audio sources, use `streaming_recognize_with_retry()` with a
restartable iterator factory.

## UI example

UI integration is intentionally omitted from the repo for now. The Python SDK
outputs `committed_text` and `unstable_text` fields that can be rendered with a
stable prefix + dimmed suffix in your UI layer.
