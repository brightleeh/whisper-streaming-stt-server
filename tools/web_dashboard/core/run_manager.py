"""Run management for the Whisper Ops web dashboard."""

from __future__ import annotations

import atexit
import csv
import hashlib
import json
import logging
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Protocol
from urllib import request as urlrequest
from urllib.error import URLError

import yaml

LOGGER = logging.getLogger("web_dashboard")

SESSION_CSV_HEADERS = [
    "session_id",
    "start_time",
    "end_time",
    "responses",
    "success",
    "error_code",
    "final_text",
    "failure_stage",
    "send_duration_seconds",
    "first_response_seconds",
    "tail_duration_seconds",
    "total_duration_seconds",
    "audio_seconds",
    "rtf",
    "decode_buffer_wait_seconds",
    "decode_queue_wait_seconds",
    "decode_inference_seconds",
    "decode_response_emit_seconds",
    "decode_total_seconds",
    "decode_count",
]


class CSVWriter(Protocol):
    def writerow(self, row: Iterable[Any], /) -> Any: ...


@dataclass(frozen=True)
class TargetConfig:
    id: str
    grpc_target: str
    http_base: str


@dataclass(frozen=True)
class RunConfig:
    target_id: str
    grpc_target: str
    http_base: str
    audio_path: str
    channels: int
    duration_sec: Optional[float]
    ramp_steps: int
    ramp_interval_sec: float
    chunk_ms: int
    realtime: bool
    speed: float
    task: str
    language: Optional[str]
    decode_profile: str
    vad_mode: str
    attrs: Dict[str, str]
    metadata: Dict[str, str]
    token: str
    create_session_backoff_ms: int
    started_at: float
    iterations: int


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    session_log_path: Path
    session_csv_path: Path
    stdout_path: Path
    stderr_path: Path
    series_path: Path
    summary_path: Path
    report_path: Path


@dataclass
class SessionRecord:
    ts: float
    success: bool
    latency_sec: Optional[float]
    rtf: Optional[float]
    audio_sec: Optional[float]
    error_code: Optional[str]


class Broadcast:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: List[queue.Queue] = []

    def subscribe(self, maxsize: int = 200) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def send(self, event: str, payload: Dict[str, Any]) -> None:
        message = {"event": event, "data": payload}
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                q.put_nowait(message)
            except queue.Full:
                # Drop on backpressure.
                pass


class RunProcess:
    def __init__(self, popen: Optional[subprocess.Popen], pid: int, pgid: int) -> None:
        self._popen = popen
        self.pid = pid
        self.pgid = pgid

    def is_running(self) -> bool:
        if self._popen is not None:
            return self._popen.poll() is None
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    def poll(self) -> Optional[int]:
        if self._popen is None:
            return None
        return self._popen.poll()

    def terminate_group(self, grace_sec: float = 3.0) -> None:
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + max(0.0, grace_sec)
        while time.time() < deadline:
            if not self.is_running():
                return
            time.sleep(0.1)
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            return


@dataclass
class RunState:
    run_id: str
    config: RunConfig
    artifacts: RunArtifacts
    process: RunProcess
    phase: str
    stop_event: threading.Event
    records: Deque[SessionRecord] = field(default_factory=deque)
    error_counts: Counter = field(default_factory=Counter)
    transcript_counts: Counter = field(default_factory=Counter)
    transcript_total: int = 0
    last_resource: Dict[str, Any] = field(default_factory=dict)
    last_runtime: Dict[str, Any] = field(default_factory=dict)
    last_metrics_ok: bool = False
    last_system_ok: bool = False
    last_grpc_ok: bool = False
    series_lock: threading.Lock = field(default_factory=threading.Lock)


class RunManager:
    def __init__(
        self,
        base_dir: Path,
        tick_interval_sec: float = 1.0,
        poll_interval_sec: float = 2.0,
        window_sec: float = 60.0,
    ) -> None:
        self.base_dir = base_dir
        self.repo_root = self._find_repo_root()
        self.runs_dir = base_dir / "runs"
        self.upload_dir = base_dir / "uploads"
        self.targets_path = base_dir / "targets.json"
        self.active_run_path = self.runs_dir / "active_run.json"
        self.tick_interval_sec = tick_interval_sec
        self.poll_interval_sec = poll_interval_sec
        self.window_sec = window_sec
        self._lock = threading.Lock()
        self._broadcast = Broadcast()
        self._state: Optional[RunState] = None

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.cleanup)
        self._restore_active_run()

    def _find_repo_root(self) -> Path:
        for candidate in (self.base_dir, *self.base_dir.parents):
            if (candidate / "config" / "model.yaml").exists():
                return candidate
        return self.base_dir

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = yaml.safe_load(path.read_text())
        except (OSError, ValueError, yaml.YAMLError) as exc:
            LOGGER.warning("Failed to load yaml %s: %s", path, exc)
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def load_defaults(self) -> Dict[str, Any]:
        file_cfg = self._load_yaml(
            self.repo_root / "stt_client" / "config" / "file.yaml"
        )
        mic_cfg = self._load_yaml(self.repo_root / "stt_client" / "config" / "mic.yaml")
        model_cfg = self._load_yaml(self.repo_root / "config" / "model.yaml")
        server_cfg = self._load_yaml(self.repo_root / "config" / "server.yaml")

        model_section = model_cfg.get("model") or {}
        server_section = server_cfg.get("server") or {}

        def pick(key: str, *sources: Dict[str, Any]) -> Any:
            for source in sources:
                if isinstance(source, dict) and key in source:
                    value = source.get(key)
                    if value is not None:
                        return value
            return None

        def as_int(value: Any, default: int) -> int:
            if isinstance(value, (int, float)):
                return int(value)
            return default

        def as_float(value: Any, default: float) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            return default

        channels = as_int(pick("max_sessions", server_section), 50)
        duration_sec = as_float(pick("session_timeout_sec", server_section), 60.0)
        chunk_ms = as_int(pick("chunk_ms", file_cfg, mic_cfg), 20)
        realtime = pick("realtime", file_cfg, mic_cfg)
        if not isinstance(realtime, bool):
            realtime = True

        task = pick("task", file_cfg, mic_cfg, model_section) or "transcribe"
        decode_profile = (
            pick("decode_profile", file_cfg, mic_cfg)
            or model_section.get("default_decode_profile")
            or "realtime"
        )
        vad_mode = pick("vad_mode", file_cfg, mic_cfg) or "auto"
        language = pick("language", file_cfg, mic_cfg, model_section) or ""
        audio_path_value = pick("audio_path", file_cfg)
        audio_candidates: List[Path] = []
        if isinstance(audio_path_value, str) and audio_path_value:
            audio_path = Path(audio_path_value)
            if audio_path.is_absolute():
                audio_candidates.append(audio_path)
            else:
                audio_candidates.append(self.repo_root / audio_path)
        default_audio = self.repo_root / "stt_client" / "assets" / "hello.wav"
        audio_candidates.append(default_audio)
        resolved_audio: Optional[Path] = None
        for candidate in audio_candidates:
            if candidate.exists():
                resolved_audio = candidate
                break

        payload = {
            "channels": channels,
            "duration_sec": duration_sec,
            "ramp_steps": 5,
            "ramp_interval_sec": 2.0,
            "chunk_ms": chunk_ms,
            "realtime": realtime,
            "speed": 1.0,
            "task": task,
            "decode_profile": decode_profile,
            "vad_mode": vad_mode,
            "language": language,
        }
        if resolved_audio is not None:
            payload["audio_path"] = str(resolved_audio)
            payload["audio_name"] = resolved_audio.name
        return payload

    def load_supported_languages(self) -> List[Dict[str, str]]:
        anchors = [self.repo_root, Path.cwd()]
        candidates: List[Path] = []
        seen: set[Path] = set()
        for anchor in anchors:
            for parent in (anchor, *anchor.parents):
                candidate = parent / "config" / "data" / "supported_languages.csv"
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if not path:
            LOGGER.warning(
                "supported_languages.csv not found in %s",
                [str(candidate) for candidate in candidates],
            )
            return []
        languages: List[Dict[str, str]] = []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header and len(header) < 2:
                    header = None
                if header is None:
                    handle.seek(0)
                    reader = csv.reader(handle)
                for row in reader:
                    if len(row) < 2:
                        continue
                    code = row[0].strip()
                    name = row[1].strip()
                    if code:
                        languages.append({"code": code, "name": name})
        except OSError as exc:
            LOGGER.warning("Failed to load languages %s: %s", path, exc)
        return languages

    def get_defaults_payload(self) -> Dict[str, Any]:
        return {
            "defaults": self.load_defaults(),
            "languages": self.load_supported_languages(),
        }

    def _restore_active_run(self) -> None:
        if not self.active_run_path.exists():
            return
        try:
            payload = json.loads(self.active_run_path.read_text())
        except (OSError, ValueError, TypeError):
            return
        run_id = payload.get("run_id")
        pid = payload.get("pid")
        pgid = payload.get("pgid")
        config_payload = payload.get("config") or {}
        artifacts_payload = payload.get("artifacts") or {}
        if not run_id or not pid or not pgid:
            return
        if "language" not in config_payload:
            config_payload["language"] = None
        if "metadata" not in config_payload:
            config_payload["metadata"] = {}
        if "create_session_backoff_ms" not in config_payload:
            config_payload["create_session_backoff_ms"] = 100
        process = RunProcess(None, int(pid), int(pgid))
        if not process.is_running():
            self._clear_active_run()
            return
        try:
            config = RunConfig(**config_payload)
        except TypeError:
            self._clear_active_run()
            return
        try:
            artifacts = RunArtifacts(
                run_dir=Path(artifacts_payload["run_dir"]),
                config_path=Path(artifacts_payload["config_path"]),
                session_log_path=Path(artifacts_payload["session_log_path"]),
                session_csv_path=Path(
                    artifacts_payload.get(
                        "session_csv_path",
                        Path(artifacts_payload["run_dir"]) / "sessions.csv",
                    )
                ),
                stdout_path=Path(artifacts_payload["stdout_path"]),
                stderr_path=Path(artifacts_payload["stderr_path"]),
                series_path=Path(artifacts_payload["series_path"]),
                summary_path=Path(artifacts_payload["summary_path"]),
                report_path=Path(artifacts_payload["report_path"]),
            )
        except (KeyError, TypeError, ValueError):
            self._clear_active_run()
            return
        state = RunState(
            run_id=run_id,
            config=config,
            artifacts=artifacts,
            process=process,
            phase="STEADY",
            stop_event=threading.Event(),
        )
        with self._lock:
            self._state = state
        self._start_workers(state, start_at_end=True)

    def list_targets(self) -> List[TargetConfig]:
        if not self.targets_path.exists():
            return []
        try:
            raw = json.loads(self.targets_path.read_text())
        except (OSError, ValueError, TypeError):
            return []
        targets: List[TargetConfig] = []
        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                if not entry.get("id") or not entry.get("grpc_target"):
                    continue
                http_base = entry.get("http_base") or ""
                targets.append(
                    TargetConfig(
                        id=str(entry["id"]),
                        grpc_target=str(entry["grpc_target"]),
                        http_base=str(http_base),
                    )
                )
        return targets

    def get_target(self, target_id: str) -> Optional[TargetConfig]:
        for target in self.list_targets():
            if target.id == target_id:
                return target
        return None

    def probe_target(self, target: TargetConfig) -> Dict[str, Any]:
        start = time.time()
        grpc_ok = self._check_grpc_target(target.grpc_target)
        system_ok, system_payload = self._fetch_json(f"{target.http_base}/system")
        metrics_ok, _ = self._fetch_json(f"{target.http_base}/metrics.json")
        rtt_ms = int((time.time() - start) * 1000)
        last_ok_ts = time.time() if (grpc_ok or system_ok or metrics_ok) else None
        runtime_payload = self._extract_runtime(system_payload)
        return {
            "grpc_ok": grpc_ok,
            "system_ok": system_ok,
            "metrics_ok": metrics_ok,
            "rtt_ms": rtt_ms,
            "ts": time.time(),
            "last_ok_ts": last_ok_ts,
            "runtime": runtime_payload or None,
        }

    def upload_audio(self, filename: str, data: bytes) -> Dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
        dest = self.upload_dir / safe_name
        dest.write_bytes(data)
        return {"audio_path": str(dest), "sha256": digest, "bytes": len(data)}

    def active_run(self) -> Optional[RunState]:
        with self._lock:
            return self._state

    def get_latest_payload(self) -> Dict[str, Any]:
        state = self.active_run()
        if not state:
            return {"active": False}
        return {
            "active": True,
            "run_id": state.run_id,
            "started_at": state.config.started_at,
            "target_id": state.config.target_id,
            "paths": {
                "run_dir": str(state.artifacts.run_dir),
                "session_log": str(state.artifacts.session_log_path),
                "session_csv": str(state.artifacts.session_csv_path),
                "stdout": str(state.artifacts.stdout_path),
                "stderr": str(state.artifacts.stderr_path),
            },
        }

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        if not self.runs_dir.exists():
            return runs
        candidates = sorted(
            [path for path in self.runs_dir.iterdir() if path.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in candidates[:limit]:
            config_path = run_dir / "config.json"
            summary_path = run_dir / "summary.json"
            config_payload: Dict[str, Any] = {}
            summary_payload: Dict[str, Any] = {}
            if config_path.exists():
                try:
                    config_payload = json.loads(config_path.read_text())
                except (OSError, ValueError, TypeError):
                    config_payload = {}
            if summary_path.exists():
                try:
                    summary_payload = json.loads(summary_path.read_text())
                except (OSError, ValueError, TypeError):
                    summary_payload = {}
            runs.append(
                {
                    "run_id": run_dir.name,
                    "config": config_payload,
                    "summary": summary_payload,
                    "paths": {
                        "run_dir": str(run_dir),
                        "session_log": str(run_dir / "sessions.jsonl"),
                        "session_csv": str(run_dir / "sessions.csv"),
                        "stdout": str(run_dir / "stdout.log"),
                        "stderr": str(run_dir / "stderr.log"),
                    },
                }
            )
        return runs

    def get_sessions_preview(self, run_id: str, limit: int = 200) -> Dict[str, Any]:
        path = self.runs_dir / run_id / "sessions.csv"
        if not path.exists():
            return {"header": [], "rows": []}
        header: List[str] = []
        rows: Deque[List[str]] = deque(maxlen=max(1, limit))
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header_row = next(reader, None)
                if header_row:
                    header = list(header_row)
                for row in reader:
                    rows.append(list(row))
        except (OSError, csv.Error) as exc:
            LOGGER.warning("Failed to read sessions.csv %s: %s", run_id, exc)
            return {"header": header, "rows": list(rows)}
        return {"header": header, "rows": list(rows)}

    def ensure_report(self, run_id: str) -> Optional[Path]:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None
        report_path = run_dir / "report.md"
        if report_path.exists():
            return report_path
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            return None
        try:
            config_payload = json.loads(config_path.read_text())
        except (OSError, ValueError, TypeError):
            config_payload = {}
        try:
            summary_payload = json.loads(summary_path.read_text())
        except (OSError, ValueError, TypeError):
            summary_payload = {}
        if not config_payload or not summary_payload:
            return None
        report = self._render_report(run_id, config_payload, summary_payload)
        try:
            report_path.write_text(report, encoding="utf-8")
        except OSError:
            return None
        return report_path

    def start_run(self, request: Dict[str, Any]) -> RunState:
        with self._lock:
            if self._state and self._state.process.is_running():
                raise RuntimeError("run already active")

        target = self.get_target(request["target_id"])
        if not target:
            raise RuntimeError("unknown target")

        run_id = uuid.uuid4().hex
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        audio_path = str(request["audio_path"])

        duration_sec = request.get("duration_sec")
        if duration_sec is not None:
            try:
                duration_sec = float(duration_sec)
            except (TypeError, ValueError):
                duration_sec = None

        iterations = int(request.get("iterations") or 1)
        if duration_sec and duration_sec > 0:
            iterations = max(iterations, 1000000)

        backoff_raw = request.get("create_session_backoff_ms")
        try:
            create_session_backoff_ms = 100 if backoff_raw is None else int(backoff_raw)
        except (TypeError, ValueError):
            create_session_backoff_ms = 100

        config = RunConfig(
            target_id=target.id,
            grpc_target=target.grpc_target,
            http_base=target.http_base,
            audio_path=audio_path,
            channels=int(request.get("channels") or 1),
            duration_sec=duration_sec,
            ramp_steps=int(request.get("ramp_steps") or 1),
            ramp_interval_sec=float(request.get("ramp_interval_sec") or 0.0),
            chunk_ms=int(request.get("chunk_ms") or 100),
            realtime=bool(request.get("realtime", True)),
            speed=float(request.get("speed") or 1.0),
            task=str(request.get("task") or "transcribe"),
            language=request.get("language") or None,
            decode_profile=str(request.get("decode_profile") or "realtime"),
            vad_mode=str(request.get("vad_mode") or "continue"),
            attrs=dict(request.get("attrs") or {}),
            metadata=dict(request.get("metadata") or {}),
            token=str(request.get("token") or ""),
            create_session_backoff_ms=create_session_backoff_ms,
            started_at=started_at,
            iterations=iterations,
        )

        artifacts = RunArtifacts(
            run_dir=run_dir,
            config_path=run_dir / "config.json",
            session_log_path=run_dir / "sessions.jsonl",
            session_csv_path=run_dir / "sessions.csv",
            stdout_path=run_dir / "stdout.log",
            stderr_path=run_dir / "stderr.log",
            series_path=run_dir / "series.jsonl",
            summary_path=run_dir / "summary.json",
            report_path=run_dir / "report.md",
        )

        artifacts.config_path.write_text(json.dumps(config.__dict__, indent=2))

        cmd = self._build_command(config, artifacts)
        stdout_handle = artifacts.stdout_path.open("w", encoding="utf-8")
        stderr_handle = artifacts.stderr_path.open("w", encoding="utf-8")

        process = subprocess.Popen(
            cmd,
            cwd=str(self.base_dir.parent.parent),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(process.pid)
        except OSError:
            pgid = process.pid
        run_process = RunProcess(process, process.pid, pgid)

        state = RunState(
            run_id=run_id,
            config=config,
            artifacts=artifacts,
            process=run_process,
            phase="RAMPING",
            stop_event=threading.Event(),
        )

        with self._lock:
            self._state = state

        self._write_active_run(state)
        self._start_workers(state, start_at_end=False)
        self._broadcast.send(
            "log",
            {
                "ts": time.time(),
                "level": "INFO",
                "source": "controller",
                "msg": f"Run started (id={run_id}, target={target.id}, channels={config.channels})",
            },
        )
        if duration_sec and duration_sec > 0:
            threading.Thread(
                target=self._auto_stop,
                args=(state, duration_sec),
                daemon=True,
            ).start()
        return state

    def stop_run(self, run_id: str, reason: str = "stopped") -> None:
        state = self.active_run()
        if not state or state.run_id != run_id:
            return
        if not state.stop_event.is_set():
            state.stop_event.set()
            state.process.terminate_group()
            self._broadcast.send(
                "log",
                {
                    "ts": time.time(),
                    "level": "WARN",
                    "source": "controller",
                    "msg": f"Run stop requested (id={run_id})",
                },
            )
        self._finalize_run(state, reason)

    def cleanup(self) -> None:
        state = self.active_run()
        if not state:
            return
        if state.process.is_running():
            state.process.terminate_group()
        self._finalize_run(state, "controller_exit")

    def subscribe(self) -> queue.Queue:
        return self._broadcast.subscribe()

    def unsubscribe(self, q: queue.Queue) -> None:
        self._broadcast.unsubscribe(q)

    def _auto_stop(self, state: RunState, duration_sec: float) -> None:
        if state.stop_event.wait(timeout=duration_sec):
            return
        self.stop_run(state.run_id, reason="finished")

    def _start_workers(self, state: RunState, start_at_end: bool) -> None:
        threading.Thread(
            target=self._tail_session_logs,
            args=(state, start_at_end),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._tail_process_log,
            args=(
                state,
                state.artifacts.stdout_path,
                "load_test",
                "INFO",
                start_at_end,
            ),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._tail_process_log,
            args=(
                state,
                state.artifacts.stderr_path,
                "load_test",
                "ERROR",
                start_at_end,
            ),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._poll_resources, args=(state,), daemon=True
        ).start()
        threading.Thread(target=self._tick_loop, args=(state,), daemon=True).start()
        threading.Thread(target=self._monitor_run, args=(state,), daemon=True).start()

    def _monitor_run(self, state: RunState) -> None:
        while not state.stop_event.is_set():
            if not state.process.is_running():
                break
            time.sleep(0.5)
        if not state.stop_event.is_set():
            state.stop_event.set()
        self._finalize_run(state, "finished")

    def _tail_session_logs(self, state: RunState, start_at_end: bool) -> None:
        path = state.artifacts.session_log_path
        handle: Optional[Any] = None
        inode: Optional[int] = None
        csv_handle: Optional[Any] = None
        csv_writer: Optional[CSVWriter] = None
        if start_at_end and path.exists():
            try:
                handle = path.open("r", encoding="utf-8")
                handle.seek(0, os.SEEK_END)
                inode = path.stat().st_ino
            except OSError:
                handle = None

        while not state.stop_event.is_set():
            if handle is None:
                if not path.exists():
                    time.sleep(0.1)
                    continue
                try:
                    handle = path.open("r", encoding="utf-8")
                    inode = path.stat().st_ino
                except OSError:
                    handle = None
                    time.sleep(0.2)
                    continue
            line = handle.readline()
            if line:
                payload = self._process_session_line(state, line)
                if payload is not None:
                    if csv_writer is None:
                        try:
                            csv_handle = state.artifacts.session_csv_path.open(
                                "a", encoding="utf-8", newline=""
                            )
                            writer = csv.writer(csv_handle)
                            csv_writer = writer
                            if state.artifacts.session_csv_path.stat().st_size == 0:
                                writer.writerow(SESSION_CSV_HEADERS)
                        except OSError:
                            csv_writer = None
                    if csv_writer is not None:
                        csv_writer.writerow(self._session_csv_row(payload))
                        if csv_handle:
                            csv_handle.flush()
                continue
            time.sleep(0.1)
            try:
                if path.exists() and inode is not None:
                    current_inode = path.stat().st_ino
                    if current_inode != inode:
                        handle.close()
                        handle = None
            except OSError:
                handle = None

        if handle:
            handle.close()
        if csv_handle:
            csv_handle.close()

    def _tail_process_log(
        self,
        state: RunState,
        path: Path,
        source: str,
        level: str,
        start_at_end: bool,
    ) -> None:
        handle: Optional[Any] = None
        if start_at_end and path.exists():
            try:
                handle = path.open("r", encoding="utf-8")
                handle.seek(0, os.SEEK_END)
            except OSError:
                handle = None

        while not state.stop_event.is_set():
            if handle is None:
                if not path.exists():
                    time.sleep(0.1)
                    continue
                try:
                    handle = path.open("r", encoding="utf-8")
                except OSError:
                    handle = None
                    time.sleep(0.2)
                    continue
            line = handle.readline()
            if line:
                message = line.rstrip()
                if message:
                    self._broadcast.send(
                        "log",
                        {
                            "ts": time.time(),
                            "level": level,
                            "source": source,
                            "msg": message,
                        },
                    )
                continue
            time.sleep(0.2)

        if handle:
            handle.close()

    def _process_session_line(
        self, state: RunState, line: str
    ) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(line)
        except ValueError:
            self._broadcast.send(
                "log",
                {
                    "ts": time.time(),
                    "level": "WARN",
                    "source": "tail",
                    "msg": "Invalid JSONL session log line skipped",
                },
            )
            return None
        success = bool(payload.get("success"))
        latency = payload.get("total_duration_seconds")
        rtf = payload.get("rtf")
        audio_sec = payload.get("audio_seconds")
        error_code = payload.get("error_code") if not success else None
        final_text = payload.get("final_text") if success else None
        record = SessionRecord(
            ts=time.time(),
            success=success,
            latency_sec=latency if isinstance(latency, (int, float)) else None,
            rtf=rtf if isinstance(rtf, (int, float)) else None,
            audio_sec=audio_sec if isinstance(audio_sec, (int, float)) else None,
            error_code=str(error_code) if error_code else None,
        )
        with state.series_lock:
            state.records.append(record)
            if record.error_code:
                state.error_counts[record.error_code] += 1
            if isinstance(final_text, str):
                normalized = self._normalize_transcript(final_text)
                if normalized:
                    state.transcript_counts[normalized] += 1
                    state.transcript_total += 1
            self._prune_records(state)
        return payload

    def _prune_records(self, state: RunState) -> None:
        cutoff = time.time() - self.window_sec
        while state.records and state.records[0].ts < cutoff:
            old = state.records.popleft()
            if old.error_code:
                state.error_counts[old.error_code] = max(
                    0, state.error_counts[old.error_code] - 1
                )

    def _poll_resources(self, state: RunState) -> None:
        while not state.stop_event.is_set():
            system_ok, system_payload = self._fetch_json(
                f"{state.config.http_base}/system"
            )
            metrics_ok, metrics_payload = self._fetch_json(
                f"{state.config.http_base}/metrics.json"
            )
            state.last_system_ok = system_ok
            state.last_metrics_ok = metrics_ok
            state.last_grpc_ok = self._check_grpc_target(state.config.grpc_target)
            if system_payload:
                state.last_resource = self._extract_resource(system_payload)
                runtime_payload = self._extract_runtime(system_payload)
                if runtime_payload:
                    state.last_runtime = runtime_payload
            time.sleep(self.poll_interval_sec)

    def _fetch_json(self, url: str) -> tuple[bool, Dict[str, Any]]:
        if not url:
            return False, {}
        try:
            with urlrequest.urlopen(url, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return True, payload if isinstance(payload, dict) else {}
        except (URLError, ValueError, TimeoutError):
            return False, {}

    def _check_grpc_target(self, target: str) -> bool:
        if not target or ":" not in target:
            return False
        host, _, port_str = target.rpartition(":")
        try:
            port = int(port_str)
        except ValueError:
            return False
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
            return True
        except OSError:
            return False

    def _extract_resource(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        process = payload.get("process") or {}
        cpu_pct = process.get("cpu_percent")
        rss_bytes = process.get("rss_bytes")
        mem_gb = None
        if isinstance(rss_bytes, (int, float)):
            mem_gb = round(rss_bytes / (1024**3), 3)
        gpu_util = None
        vram_gb = None
        gpus = payload.get("gpus") or []
        if isinstance(gpus, list) and gpus:
            gpu = gpus[0]
            gpu_util = gpu.get("utilization_gpu")
            mem_used = gpu.get("memory_used_mib")
            if isinstance(mem_used, (int, float)):
                vram_gb = round(mem_used / 1024.0, 3)
        return {
            "ts": time.time(),
            "cpu_pct": cpu_pct if isinstance(cpu_pct, (int, float)) else None,
            "mem_gb": mem_gb,
            "gpu_util_pct": gpu_util if isinstance(gpu_util, (int, float)) else None,
            "vram_gb": vram_gb,
        }

    def _extract_runtime(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        runtime = payload.get("runtime")
        if isinstance(runtime, dict):
            return runtime
        return {}

    def _tick_loop(self, state: RunState) -> None:
        while not state.stop_event.is_set():
            self._emit_tick(state)
            time.sleep(self.tick_interval_sec)

    def _emit_tick(self, state: RunState) -> None:
        now = time.time()
        elapsed = max(0.0, now - state.config.started_at)
        target = state.config.channels
        active = self._estimate_active(state.config, elapsed)
        phase = self._phase_for(state, elapsed)

        with state.series_lock:
            latencies = [r.latency_sec for r in state.records if r.latency_sec]
            rtfs = [r.rtf for r in state.records if r.rtf]
            audio_secs = [r.audio_sec for r in state.records if r.audio_sec]
            total_count = len(state.records)
            error_count = sum(1 for r in state.records if not r.success)
            error_rate = (error_count / total_count) if total_count else 0.0
            error_per_sec = error_count / max(1.0, self.window_sec)
            top_codes = [
                {"code": code, "count": count}
                for code, count in state.error_counts.most_common(3)
            ]
        latency_ms = self._percentiles_ms(latencies)
        rtf_vals = self._percentiles(rtfs)
        throughput = self._throughput(audio_secs)

        kpi = {
            "ts": now,
            "phase": phase,
            "active": active,
            "target": target,
            "latency_ms": latency_ms,
            "rtf": rtf_vals,
            "throughput_audio_sps": throughput,
            "errors": {
                "count": error_count,
                "rate": error_rate,
                "per_sec": error_per_sec,
                "top_codes": top_codes,
            },
        }
        resource = (
            dict(state.last_resource)
            if state.last_resource
            else {
                "ts": now,
                "cpu_pct": None,
                "mem_gb": None,
                "gpu_util_pct": None,
                "vram_gb": None,
            }
        )
        if state.last_runtime:
            resource["runtime"] = state.last_runtime
        resource.update(
            {
                "system_ok": state.last_system_ok,
                "metrics_ok": state.last_metrics_ok,
                "grpc_ok": state.last_grpc_ok,
            }
        )
        self._broadcast.send("kpi", kpi)
        self._broadcast.send("resource", resource)
        self._append_series(state, "kpi", kpi)
        self._append_series(state, "resource", resource)

    def _append_series(
        self, state: RunState, event: str, payload: Dict[str, Any]
    ) -> None:
        try:
            with state.artifacts.series_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": event, "data": payload}) + "\n")
        except OSError:
            return

    def _phase_for(self, state: RunState, elapsed: float) -> str:
        if not state.process.is_running() and state.stop_event.is_set():
            return "DONE"
        ramp_total = max(
            0.0, (state.config.ramp_steps - 1) * state.config.ramp_interval_sec
        )
        if elapsed < ramp_total:
            return "RAMPING"
        return "STEADY"

    def _estimate_active(self, config: RunConfig, elapsed: float) -> int:
        if config.ramp_steps <= 1:
            return config.channels
        step_size = int((config.channels + config.ramp_steps - 1) / config.ramp_steps)
        steps_done = int(elapsed // max(0.001, config.ramp_interval_sec)) + 1
        return min(config.channels, step_size * steps_done)

    def _percentiles_ms(self, values: Iterable[float]) -> Dict[str, float]:
        data = sorted(values)
        return {
            "p50": self._percentile(data, 50) * 1000.0,
            "p95": self._percentile(data, 95) * 1000.0,
            "p99": self._percentile(data, 99) * 1000.0,
        }

    @staticmethod
    def _normalize_transcript(text: str) -> str:
        if not text:
            return ""
        normalized = " ".join(text.strip().split())
        return normalized.lower()

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return ""
        try:
            return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError):
            return ""

    @classmethod
    def _session_csv_row(cls, payload: Dict[str, Any]) -> List[str]:
        row: List[str] = []
        for key in SESSION_CSV_HEADERS:
            if key == "start_time":
                value = payload.get("start_time") or cls._format_timestamp(
                    payload.get("start_ts")
                )
            elif key == "end_time":
                value = payload.get("end_time") or cls._format_timestamp(
                    payload.get("end_ts")
                )
            else:
                value = payload.get(key, "")
            if value is None:
                value = ""
            if isinstance(value, bool):
                row.append("true" if value else "false")
            else:
                text = str(value)
                row.append(text.replace("\n", " ").replace("\r", " "))
        return row

    def _percentiles(self, values: Iterable[float]) -> Dict[str, float]:
        data = sorted(values)
        return {
            "p50": self._percentile(data, 50),
            "p95": self._percentile(data, 95),
            "p99": self._percentile(data, 99),
        }

    def _throughput(self, audio_secs: List[float]) -> float:
        if not audio_secs:
            return 0.0
        return sum(audio_secs) / max(1.0, self.window_sec)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        index = max(0, int(round((pct / 100.0) * (len(values) - 1))))
        index = min(index, len(values) - 1)
        return float(values[index])

    def _build_command(self, config: RunConfig, artifacts: RunArtifacts) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "tools.bench.grpc_load_test",
            "--target",
            config.grpc_target,
            "--audio",
            config.audio_path,
            "--channels",
            str(config.channels),
            "--iterations",
            str(config.iterations),
            "--chunk-ms",
            str(config.chunk_ms),
            "--session-log-format",
            "jsonl",
            "--session-log-path",
            str(artifacts.session_log_path),
            "--ramp-steps",
            str(max(1, config.ramp_steps)),
            "--ramp-interval-sec",
            str(max(0.0, config.ramp_interval_sec)),
            "--task",
            config.task,
            "--language",
            str(config.language or ""),
            "--decode-profile",
            config.decode_profile,
            "--vad-mode",
            config.vad_mode,
        ]
        if config.create_session_backoff_ms > 0:
            cmd.extend(
                ["--create-session-backoff-ms", str(config.create_session_backoff_ms)]
            )
        if config.realtime:
            cmd.append("--realtime")
            cmd.extend(["--speed", str(max(0.1, config.speed))])
        if config.attrs:
            for key, value in config.attrs.items():
                cmd.extend(["--attr", f"{key}={value}"])
        if config.metadata:
            for key, value in config.metadata.items():
                cmd.extend(["--metadata", f"{key}={value}"])
        if config.token:
            token_value = config.token.strip()
            if token_value:
                if " " not in token_value:
                    token_value = f"Bearer {token_value}"
                has_auth = any(
                    key.lower() == "authorization" for key in config.metadata.keys()
                )
                if not has_auth:
                    cmd.extend(["--metadata", f"authorization={token_value}"])
        return cmd

    def _write_active_run(self, state: RunState) -> None:
        payload = {
            "run_id": state.run_id,
            "pid": state.process.pid,
            "pgid": state.process.pgid,
            "config": state.config.__dict__,
            "artifacts": {
                "run_dir": str(state.artifacts.run_dir),
                "config_path": str(state.artifacts.config_path),
                "session_log_path": str(state.artifacts.session_log_path),
                "session_csv_path": str(state.artifacts.session_csv_path),
                "stdout_path": str(state.artifacts.stdout_path),
                "stderr_path": str(state.artifacts.stderr_path),
                "series_path": str(state.artifacts.series_path),
                "summary_path": str(state.artifacts.summary_path),
                "report_path": str(state.artifacts.report_path),
            },
        }
        self.active_run_path.write_text(json.dumps(payload, indent=2))

    def _clear_active_run(self) -> None:
        try:
            self.active_run_path.unlink()
        except OSError:
            pass

    def _finalize_run(self, state: RunState, reason: str) -> None:
        if state.phase == "DONE":
            return
        state.phase = "DONE"
        summary = self._build_summary(state)
        try:
            state.artifacts.summary_path.write_text(json.dumps(summary, indent=2))
        except OSError as exc:
            LOGGER.warning("Failed to write summary %s: %s", state.run_id, exc)
            self._broadcast.send(
                "log",
                {
                    "ts": time.time(),
                    "level": "WARN",
                    "source": "controller",
                    "msg": f"Failed to write summary.json: {exc}",
                },
            )
        try:
            report = self._render_report(state.run_id, state.config.__dict__, summary)
            state.artifacts.report_path.write_text(report, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Failed to write report %s: %s", state.run_id, exc)
            self._broadcast.send(
                "log",
                {
                    "ts": time.time(),
                    "level": "WARN",
                    "source": "controller",
                    "msg": f"Failed to write report.md: {exc}",
                },
            )
        try:
            summary_line = self._format_summary_line(summary)
            LOGGER.info("Run summary %s: %s", state.run_id, summary_line)
            self._broadcast.send(
                "log",
                {
                    "ts": time.time(),
                    "level": "INFO",
                    "source": "controller",
                    "msg": f"Summary: {summary_line}",
                },
            )
        except Exception as exc:  # pragma: no cover - safety net
            LOGGER.warning("Failed to emit summary log %s: %s", state.run_id, exc)
        self._broadcast.send(
            "log",
            {
                "ts": time.time(),
                "level": "INFO",
                "source": "controller",
                "msg": f"Run finished (id={state.run_id}, reason={reason})",
            },
        )
        self._broadcast.send("done", {"ts": time.time(), "reason": reason})
        self._clear_active_run()
        with self._lock:
            if self._state and self._state.run_id == state.run_id:
                self._state = None

    def _build_summary(self, state: RunState) -> Dict[str, Any]:
        with state.series_lock:
            latencies = [r.latency_sec for r in state.records if r.latency_sec]
            rtfs = [r.rtf for r in state.records if r.rtf]
            audio_secs = [r.audio_sec for r in state.records if r.audio_sec]
            total = len(state.records)
            errors = sum(1 for r in state.records if not r.success)
            transcript_total = state.transcript_total
            transcript_counts = Counter(state.transcript_counts)
        return {
            "run_id": state.run_id,
            "total_sessions": total,
            "errors": errors,
            "latency_ms": self._percentiles_ms(latencies),
            "rtf": self._percentiles(rtfs),
            "throughput_audio_sps": self._throughput(audio_secs),
            "transcripts": self._summarize_transcripts(
                transcript_counts, transcript_total
            ),
        }

    @staticmethod
    def _summarize_transcripts(counts: Counter, total: int) -> Dict[str, Any]:
        if total <= 0 or not counts:
            return {
                "total": total,
                "unique": 0,
                "match_rate": 0.0,
                "top_text": "",
            }
        top_text, top_count = counts.most_common(1)[0]
        match_rate = top_count / total if total else 0.0
        # Avoid extremely long blobs in summary/report.
        max_len = 200
        if len(top_text) > max_len:
            top_text = top_text[: max_len - 3] + "..."
        return {
            "total": total,
            "unique": len(counts),
            "match_rate": round(match_rate, 4),
            "top_text": top_text,
        }

    def _render_report(
        self, run_id: str, config: Dict[str, Any], summary: Dict[str, Any]
    ) -> str:
        started_at = config.get("started_at")
        started_text = (
            datetime.fromtimestamp(float(started_at)).isoformat(timespec="seconds")
            if started_at
            else "unknown"
        )
        total = int(summary.get("total_sessions") or 0)
        errors = int(summary.get("errors") or 0)
        error_rate = (errors / total) if total else 0.0
        latency = summary.get("latency_ms") or {}
        rtf = summary.get("rtf") or {}
        throughput = summary.get("throughput_audio_sps")
        transcripts = summary.get("transcripts") or {}
        realtime = bool(config.get("realtime", True))
        rtf_p95 = rtf.get("p95") if isinstance(rtf, dict) else None

        health = "UNKNOWN"
        if realtime:
            if isinstance(rtf_p95, (int, float)) and (
                rtf_p95 > 1.2 or error_rate > 0.05
            ):
                health = "OVERLOADED"
            elif isinstance(rtf_p95, (int, float)) and (
                rtf_p95 > 1.0 or error_rate > 0.01
            ):
                health = "AT RISK"
            else:
                health = "HEALTHY"
        else:
            health = "THROUGHPUT MODE"

        def fmt(value: Any, digits: int = 3) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.{digits}f}"
            return "--"

        def fmt_sec_from_ms(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"{value / 1000.0:.3f}s"
            return "--"

        lines = [
            "# Whisper Ops Report",
            "",
            "## Run",
            f"- Run ID: {run_id}",
            f"- Started: {started_text}",
            f"- Target: {config.get('grpc_target', '--')}",
            f"- Audio: {config.get('audio_path', '--')}",
            f"- Channels: {config.get('channels', '--')}",
            f"- Duration (sec): {config.get('duration_sec', '--')}",
            f"- Ramp: steps={config.get('ramp_steps', '--')} interval={config.get('ramp_interval_sec', '--')}",
            f"- Chunk (ms): {config.get('chunk_ms', '--')}",
            f"- Mode: {'realtime' if realtime else 'throughput'} (speed={config.get('speed', '--')})",
            f"- Decode profile: {config.get('decode_profile', '--')}",
            f"- VAD mode: {config.get('vad_mode', '--')}",
            f"- Language: {config.get('language') or 'auto'}",
            "",
            "## Summary",
            f"- Total sessions: {total}",
            f"- Errors: {errors} ({fmt(error_rate * 100, 1)}%)",
            (
                f"- Latency s (p50/p95/p99): {fmt_sec_from_ms(latency.get('p50'))}/"
                f"{fmt_sec_from_ms(latency.get('p95'))}/{fmt_sec_from_ms(latency.get('p99'))}"
            ),
            (
                f"- RTF (p50/p95/p99): {fmt(rtf.get('p50'))}/"
                f"{fmt(rtf.get('p95'))}/{fmt(rtf.get('p99'))}"
            ),
            f"- Throughput (audio-sec/s): {fmt(throughput)}",
            (
                f"- Transcript match rate: "
                f"{fmt((transcripts.get('match_rate') or 0.0) * 100, 1)}% "
                f"(unique={transcripts.get('unique', 0)}, "
                f"total={transcripts.get('total', 0)})"
            ),
            (f"- Top transcript: {transcripts.get('top_text') or '--'}"),
            "",
            "## Verdict",
            f"- Health: {health}",
            f"- RTF p95: {fmt(rtf_p95)}",
            f"- Error rate: {fmt(error_rate * 100, 1)}%",
            "",
            "_Heuristic only. Compare against baseline runs for final judgement._",
        ]
        return "\n".join(lines)

    def _format_summary_line(self, summary: Dict[str, Any]) -> str:
        latency = summary.get("latency_ms") or {}
        rtf = summary.get("rtf") or {}
        throughput = summary.get("throughput_audio_sps")
        transcripts = summary.get("transcripts") or {}
        total = int(summary.get("total_sessions") or 0)
        errors = int(summary.get("errors") or 0)
        error_rate = (errors / total) if total else 0.0

        def sec(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"{value / 1000.0:.3f}s"
            return "--"

        def num(value: Any, digits: int = 3) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.{digits}f}"
            return "--"

        parts = [
            f"sessions={total}",
            f"errors={errors} ({error_rate * 100:.1f}%)",
            f"latency p50/p95/p99={sec(latency.get('p50'))}/{sec(latency.get('p95'))}/{sec(latency.get('p99'))}",
            f"rtf p50/p95/p99={num(rtf.get('p50'), 3)}/{num(rtf.get('p95'), 3)}/{num(rtf.get('p99'), 3)}",
            f"throughput={num(throughput, 3)} audio-sec/s",
            f"match_rate={(transcripts.get('match_rate') or 0.0) * 100:.1f}%",
        ]
        return " | ".join(parts)
