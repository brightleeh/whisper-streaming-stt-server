import sys as _sys

from . import stt_pb2 as stt_pb2

# Allow "import stt_pb2" style access that some tools expect before stt_pb2_grpc loads.
_sys.modules.setdefault("stt_pb2", stt_pb2)

from . import stt_pb2_grpc as stt_pb2_grpc

_sys.modules.setdefault("stt_pb2_grpc", stt_pb2_grpc)

__all__ = ("stt_pb2", "stt_pb2_grpc")
