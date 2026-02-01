from concurrent import futures
import socket

import grpc
import pytest

from gen.stt.python.v1 import stt_pb2, stt_pb2_grpc
from stt_client.realtime.file import _create_channel as create_realtime_channel

_CERT_PEM = """-----BEGIN CERTIFICATE-----
MIICpDCCAYwCCQCHSeVShcNnMjANBgkqhkiG9w0BAQsFADAUMRIwEAYDVQQDDAls
b2NhbGhvc3QwHhcNMjYwMTMxMjI0ODIxWhcNMjcwMTMxMjI0ODIxWjAUMRIwEAYD
VQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDJ
/TmyBXcoN4t+uLnn2VuGApVHqMdELWux3XZyyMYamThZvXttpmXrYKJARixrJvd7
x0T5tvR8v/m9vNHutTNCLCP/uSvTecFIKAm567v7WVajn+2BJjKHvpNG+5jgD+Qf
/amgjHs13k3va3o/7a2xz5qZ3yGbJ1TTq5EYjPdBqjQKkrPgOHmWsOpIhrHwDf9B
JN2+2vTHeXIwUhpo3/iW5z6ppczGBmBFElfQRIgi/VAtHwhZ8yRkvN7gOO9/L3Y2
Q/Q7iMVaYMMWRcp3oo/J1jST/0AlYWA2X0BZGYnRRva2c9PkpMTsw8fQ5fUtvE3n
Jhm1KCWC4NqATPjv+FMBAgMBAAEwDQYJKoZIhvcNAQELBQADggEBAGRiMOPYGo5c
UA37p4QOrqWY3Ilm5ceBuBoNSEqqB5RnAGUsPqg3UGyhNOW8Sr5EW8jT+JC3spsw
FWf9lgw6+lDehlJ5+pa4AI/6LQAv4FFUS5ySw1lbcYIKZbFZruoUwltM6R/tgwVW
pj/PPbEXJNmBhxf/EqSdimT6hq31cdet7tcPoBdjjzqKSLTlO5JujDiB7NHmmOof
ytXbNGJYeDfhv/8cTOBVj9rXICYvcWeCVQYPb4yxs2DnN31vwiOzwgW72SqNuKhd
F66uzBxsvchGMd+owCpWUYVWFAnHgjadhO+whRhybC+2S9RW2kJkIYPtBRXeO5V0
T1XdAljs7PM=
-----END CERTIFICATE-----
"""

_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDJ/TmyBXcoN4t+
uLnn2VuGApVHqMdELWux3XZyyMYamThZvXttpmXrYKJARixrJvd7x0T5tvR8v/m9
vNHutTNCLCP/uSvTecFIKAm567v7WVajn+2BJjKHvpNG+5jgD+Qf/amgjHs13k3v
a3o/7a2xz5qZ3yGbJ1TTq5EYjPdBqjQKkrPgOHmWsOpIhrHwDf9BJN2+2vTHeXIw
Uhpo3/iW5z6ppczGBmBFElfQRIgi/VAtHwhZ8yRkvN7gOO9/L3Y2Q/Q7iMVaYMMW
Rcp3oo/J1jST/0AlYWA2X0BZGYnRRva2c9PkpMTsw8fQ5fUtvE3nJhm1KCWC4NqA
TPjv+FMBAgMBAAECggEALoR/5gy2LW/lU1DNl6SKHGm54V/wTDY9qB+9qJ+uZ+/D
d39Yzp7UuAlwNGc00ZSOyFXS/8NvuM8pf9XdbyER5fpua/VEL5bJuYlm/AdbKn6f
ol0xgF3Ao1xzZJSK14cxXWC40P2pXnWM80eIRvLA3sNV8nvdrHK5aWoIp2PHibp1
J7eR04jun2prsH18Mn4ctJpmNN6xDt7GTQsmind+u+nFD6cG/Wd5FgBuwsz55uMH
6S3A2pBG3NGiX/ZARfxpHZLnWFN8KaWu8NlZciTIjBdsAaBLlpRtNbJAPCPTnhmf
1JkP1jQCqnZ8vemaPodm7NkWnlOwnP+5sbr0r+nK8QKBgQDrEc6TQ9yS6CE234xQ
+uycUt5wO0OL65xDMpMBd5FlTMupeBngssTCFQw7RoalElHxjrx+1U2I1bH9qyBG
xq5ikO77p1Yz/cE3fduqG03eNh1csM9Kqhh1sy2G7ECIYqdc99qqqJ37+TGBjQyC
/sg6Ou+reLkfYlwpWubZ1ndVjQKBgQDb+WGjwV+C394hCKaTxkthOPRwaO3/0tw7
FnxTuICTD+Mm/Mt3Tg3TfVNbsuC7H4MJ4B3PAQpVtawhJGVFK2x71zLLgh6UMS/6
wIYKQF/XtWh6YLs1kHPZZW+Z3sKWKzbdTGX2wKau7KpbOQFR4vAd+ya3Yk2yEjJz
alp2O6xURQKBgQCYZoYQUenaUKbgBZTaF6R5QPy3tKR5PXqk0lAenl9kVqKfr57P
X/dSgQTaFUJMGRGJU2n/rNjEww7PkDevyzXZ728RNo8bzAONr4pPwb39OAZXRsZN
+PM8s7rrg5XfFl69Vm+tPv6WExw2irS25On5XqZt/CnBICryIN4UEwhxLQKBgHnk
DzmpXl5r5G9TltJNz9k4sSJU0oSueAB57jyKARz8cbdZ3vjmFH07deRbE3I8/OSM
/peFEQ/7Uj0vKLqSXFOnJGtmV8FwHBELe3rUvwcNa65cSYBd4gP11EhkClkh6w3n
VpzLldFaLO/Nf7C7WqiSUZrOaxUgRjp0FVpsqIotAoGBANV9dV9ADiMZ00LPFOQy
MIEvAf/qK5Tbk0id/Z91s1TSmo39/4CVaz19AVaeoaqrBJip8IaUDh+WgcCTLvus
lEUEPmAgLv1dJA/JY0PJTaFvVyluPD2kxC6Zjn4sp67l3xrrOGTn9WZRETsbTxZo
K1MwvTOaTymuuqcJWlGw7Z1C
-----END PRIVATE KEY-----
"""


class FakeServicer(stt_pb2_grpc.STTBackendServicer):
    """Minimal servicer used to validate TLS connectivity."""

    def CreateSession(self, request, context):  # pylint: disable=unused-argument
        return stt_pb2.SessionResponse()

    def StreamingRecognize(
        self, request_iterator, context
    ):  # pylint: disable=unused-argument
        if False:
            yield  # pragma: no cover - keeps generator semantics
        return


def _write_tls_files(tmp_path):
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    cert_path.write_text(_CERT_PEM, encoding="utf-8")
    key_path.write_text(_KEY_PEM, encoding="utf-8")
    return cert_path, key_path


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_tls_grpc_client_server_roundtrip(tmp_path):
    """Ensure client can talk to a TLS-enabled gRPC server."""
    cert_path, key_path = _write_tls_files(tmp_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    stt_pb2_grpc.add_STTBackendServicer_to_server(FakeServicer(), server)
    credentials = grpc.ssl_server_credentials(
        [(key_path.read_bytes(), cert_path.read_bytes())]
    )
    port = _pick_free_port()
    port = server.add_secure_port(f"localhost:{port}", credentials)
    assert port > 0
    server.start()

    channel = None
    try:
        channel = create_realtime_channel(
            f"localhost:{port}", None, None, True, str(cert_path)
        )
        stub = stt_pb2_grpc.STTBackendStub(channel)
        response = stub.CreateSession(stt_pb2.SessionRequest(session_id="tls-test"))
        assert isinstance(response, stt_pb2.SessionResponse)
    finally:
        if channel is not None:
            channel.close()
        server.stop(0)


def test_tls_grpc_plaintext_connection_fails(tmp_path):
    """Ensure plaintext gRPC fails against TLS-only server."""
    cert_path, key_path = _write_tls_files(tmp_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    stt_pb2_grpc.add_STTBackendServicer_to_server(FakeServicer(), server)
    credentials = grpc.ssl_server_credentials(
        [(key_path.read_bytes(), cert_path.read_bytes())]
    )
    port = _pick_free_port()
    port = server.add_secure_port(f"localhost:{port}", credentials)
    assert port > 0
    server.start()

    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = stt_pb2_grpc.STTBackendStub(channel)
            with pytest.raises(grpc.RpcError) as exc:
                stub.CreateSession(stt_pb2.SessionRequest(session_id="plain-test"))
            assert exc.value.code() in {
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.INTERNAL,
            }
    finally:
        server.stop(0)
