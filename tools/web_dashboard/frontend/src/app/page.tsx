"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import LangToggle from "./LangToggle";
import ThemeToggle from "./ThemeToggle";
import { MESSAGES, detectLocale, type Locale } from "./i18n";

const CONTROLLER_BASE =
  process.env.NEXT_PUBLIC_CONTROLLER_BASE || "http://localhost:8010";

type Target = {
  id: string;
  grpc_target: string;
  http_base: string;
};

type KPI = {
  ts: number;
  phase: string;
  active: number;
  target: number;
  latency_ms: { p50: number; p95: number; p99: number };
  rtf: { p50: number; p95: number; p99: number };
  throughput_audio_sps: number;
  errors: {
    count: number;
    rate: number;
    per_sec?: number;
    top_codes: { code: string; count: number }[];
  };
};

type Resource = {
  ts: number;
  cpu_pct: number | null;
  mem_gb: number | null;
  gpu_util_pct: number | null;
  vram_gb: number | null;
  grpc_ok?: boolean;
  system_ok?: boolean;
  metrics_ok?: boolean;
  runtime?: RuntimeInfo;
};

type RuntimeInfo = {
  model?: {
    model_size?: string;
    model_backend?: string;
    device?: string;
    compute_type?: string;
    model_pool_size?: number;
    default_decode_profile?: string;
    language?: string;
    task?: string;
  };
  streaming?: {
    sample_rate?: number;
    session_timeout_sec?: number;
    decode_timeout_sec?: number;
    create_session_rps?: number;
    create_session_burst?: number;
    vad_model_pool_size?: number;
    vad_model_prewarm?: number;
    vad_silence?: number;
    vad_threshold?: number;
    max_chunk_ms?: number;
    partial_decode_interval_sec?: number;
    partial_decode_window_sec?: number;
    decode_batch_window_ms?: number;
    max_decode_batch_size?: number;
    max_pending_decodes_global?: number;
    max_pending_decodes_per_stream?: number;
    adaptive_throttle_enabled?: boolean;
  };
};

type LogEntry = {
  ts: number;
  level: string;
  source: string;
  msg: string;
};

type SessionsPreview = {
  run_id: string | null;
  header: string[];
  rows: string[][];
};

type LanguageOption = {
  code: string;
  name: string;
};

export default function HomePage() {
  const [locale, setLocale] = useState<Locale>("en");
  const [targets, setTargets] = useState<Target[]>([]);
  const [targetId, setTargetId] = useState<string>("");
  const [audioPath, setAudioPath] = useState<string>("");
  const [inputMode, setInputMode] = useState<"file" | "mic">("file");
  const [recording, setRecording] = useState<boolean>(false);
  const [recordSecs, setRecordSecs] = useState<number>(0);
  const [recordStatus, setRecordStatus] = useState<string>("");
  const [channels, setChannels] = useState<number>(50);
  const [durationSec, setDurationSec] = useState<number>(60);
  const [rampSteps, setRampSteps] = useState<number>(5);
  const [rampInterval, setRampInterval] = useState<number>(2);
  const [chunkMs, setChunkMs] = useState<number>(20);
  const [realtime, setRealtime] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(1.0);
  const [decodeProfile, setDecodeProfile] = useState<string>("realtime");
  const [vadMode, setVadMode] = useState<string>("auto");
  const [language, setLanguage] = useState<string>("");
  const [languageOptions, setLanguageOptions] = useState<LanguageOption[]>([]);
  const [token, setToken] = useState<string>("");
  const [attrsText, setAttrsText] = useState<string>("");
  const [metadataText, setMetadataText] = useState<string>("");
  const [runId, setRunId] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [kpi, setKpi] = useState<KPI | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [targetStatus, setTargetStatus] = useState<Resource | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [compareA, setCompareA] = useState<string>("");
  const [compareB, setCompareB] = useState<string>("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [sessionsPreview, setSessionsPreview] = useState<SessionsPreview>({
    run_id: null,
    header: [],
    rows: [],
  });
  const [uploadName, setUploadName] = useState<string>("");
  const [latencySeries, setLatencySeries] = useState<any[]>([]);
  const [throughputSeries, setThroughputSeries] = useState<any[]>([]);
  const [gpuSeries, setGpuSeries] = useState<any[]>([]);
  const [hasGpu, setHasGpu] = useState<boolean>(false);
  const [runtimeSnapshot, setRuntimeSnapshot] = useState<RuntimeInfo | null>(null);
  const [runtimeTs, setRuntimeTs] = useState<number | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordChunksRef = useRef<Blob[]>([]);
  const recordTimerRef = useRef<number | null>(null);
  const dict = MESSAGES[locale] ?? MESSAGES.en;
  const t = (key: string) => dict[key] ?? MESSAGES.en[key] ?? key;

  useEffect(() => {
    let next: Locale = "en";
    try {
      const stored = window.localStorage.getItem("wops-locale");
      if (stored === "en" || stored === "ko" || stored === "ja") {
        next = stored;
      } else {
        next = detectLocale(window.navigator.language);
      }
    } catch {
      next = "en";
    }
    setLocale(next);
    document.documentElement.lang = next;
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("wops-locale", locale);
    } catch {
      // ignore
    }
    document.documentElement.lang = locale;
  }, [locale]);

  useEffect(() => {
    fetch(`${CONTROLLER_BASE}/targets`)
      .then((res) => res.json())
      .then((data) => {
        setTargets(data);
      })
      .catch(() => setTargets([]));
  }, []);

  useEffect(() => {
    if (!targetId && targets.length > 0) {
      setTargetId(targets[0].id);
    }
  }, [targetId, targets]);

  useEffect(() => {
    fetch(`${CONTROLLER_BASE}/defaults`)
      .then((res) => res.json())
      .then((data) => {
        const defaults = data.defaults || {};
        if (typeof defaults.channels === "number") setChannels(defaults.channels);
        if (typeof defaults.duration_sec === "number")
          setDurationSec(defaults.duration_sec);
        if (typeof defaults.ramp_steps === "number") setRampSteps(defaults.ramp_steps);
        if (typeof defaults.ramp_interval_sec === "number")
          setRampInterval(defaults.ramp_interval_sec);
        if (typeof defaults.chunk_ms === "number") setChunkMs(defaults.chunk_ms);
        if (typeof defaults.realtime === "boolean") setRealtime(defaults.realtime);
        if (typeof defaults.speed === "number") setSpeed(defaults.speed);
        if (typeof defaults.decode_profile === "string")
          setDecodeProfile(defaults.decode_profile);
        if (typeof defaults.vad_mode === "string") setVadMode(defaults.vad_mode);
        if (typeof defaults.language === "string") setLanguage(defaults.language);
        if (typeof defaults.audio_path === "string" && defaults.audio_path) {
          const name =
            typeof defaults.audio_name === "string" && defaults.audio_name
              ? defaults.audio_name
              : defaults.audio_path.split("/").pop() || defaults.audio_path;
          setAudioPath((prev) => prev || defaults.audio_path);
          setUploadName((prev) => prev || name);
        }
        if (Array.isArray(data.languages)) setLanguageOptions(data.languages);
      })
      .catch(() => undefined);
  }, []);

  const refreshHistory = () => {
    fetch(`${CONTROLLER_BASE}/runs/history`)
      .then((res) => res.json())
      .then((data) => setHistory(data.runs || []))
      .catch(() => setHistory([]));
  };

  useEffect(() => {
    refreshHistory();
  }, []);

  const previewRunId = useMemo(() => {
    if (runId) return runId;
    if (lastRunId) return lastRunId;
    return null;
  }, [runId, lastRunId]);

  useEffect(() => {
    const id = previewRunId;
    if (!id) {
      setSessionsPreview({ run_id: null, header: [], rows: [] });
      return;
    }
    let active = true;
    const fetchPreview = () => {
      fetch(`${CONTROLLER_BASE}/runs/${id}/sessions/preview?limit=200`)
        .then((res) => res.json())
        .then((data) => {
          if (!active) return;
          const header = Array.isArray(data.header) ? data.header : [];
          const rows = Array.isArray(data.rows) ? data.rows : [];
          setSessionsPreview({ run_id: id, header, rows });
        })
        .catch(() => {
          if (!active) return;
          setSessionsPreview((prev) =>
            prev.run_id === id ? prev : { run_id: id, header: [], rows: [] }
          );
        });
    };
    fetchPreview();
    const interval = window.setInterval(fetchPreview, runId ? 2000 : 6000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [previewRunId, runId]);

  useEffect(() => {
    if (!targetId) return;
    setRuntimeSnapshot(null);
    setRuntimeTs(null);
    const fetchStatus = () => {
      fetch(`${CONTROLLER_BASE}/targets/${targetId}/status`)
        .then((res) => res.json())
        .then((data) => {
          setTargetStatus({
            ts: data.ts,
            cpu_pct: null,
            mem_gb: null,
            gpu_util_pct: null,
            vram_gb: null,
            grpc_ok: data.grpc_ok,
            system_ok: data.system_ok,
            metrics_ok: data.metrics_ok,
            runtime: data.runtime,
          });
          if (data.runtime) {
            setRuntimeSnapshot(data.runtime);
            setRuntimeTs(typeof data.ts === "number" ? data.ts : Date.now() / 1000);
          }
        })
        .catch(() => setTargetStatus(null));
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [targetId]);

  useEffect(() => {
    fetch(`${CONTROLLER_BASE}/runs/latest`)
      .then((res) => res.json())
      .then((data) => {
        if (data.active && data.run_id) {
          setRunId(data.run_id);
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (inputMode !== "mic" && recording) {
      stopRecording();
    }
  }, [inputMode, recording]);

  useEffect(() => {
    if (!runId) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    setLastRunId(runId);

    const source = new EventSource(`${CONTROLLER_BASE}/runs/${runId}/live`);
    eventSourceRef.current = source;

    source.addEventListener("kpi", (event) => {
      const payload: KPI = JSON.parse((event as MessageEvent).data);
      setKpi(payload);
      setLatencySeries((prev) =>
        appendSeries(prev, {
          ts: payload.ts,
          p95: (payload.latency_ms.p95 ?? 0) / 1000,
          p99: (payload.latency_ms.p99 ?? 0) / 1000,
        })
      );
      setThroughputSeries((prev) =>
        appendSeries(prev, {
          ts: payload.ts,
          active: payload.active,
          throughput: payload.throughput_audio_sps,
        })
      );
    });

    source.addEventListener("resource", (event) => {
      const payload: Resource = JSON.parse((event as MessageEvent).data);
      setResource(payload);
      if (payload.runtime) {
        setRuntimeSnapshot(payload.runtime);
        setRuntimeTs(typeof payload.ts === "number" ? payload.ts : Date.now() / 1000);
      }
      const hasGpuMetrics =
        payload.gpu_util_pct !== null || payload.vram_gb !== null;
      if (hasGpuMetrics) {
        setHasGpu(true);
        setGpuSeries((prev) =>
          appendSeries(prev, {
            ts: payload.ts,
            gpu: payload.gpu_util_pct ?? null,
            vram: payload.vram_gb ?? null,
          })
        );
      }
    });

    source.addEventListener("log", (event) => {
      const payload: LogEntry = JSON.parse((event as MessageEvent).data);
      setLogs((prev) => appendLogs(prev, payload));
    });

    source.addEventListener("done", () => {
      source.close();
      eventSourceRef.current = null;
      setRunId(null);
      refreshHistory();
    });

    source.onerror = () => {
      setLogs((prev) =>
        appendLogs(prev, {
          ts: Date.now() / 1000,
          level: "WARN",
          source: "sse",
          msg: t("sse_dropped"),
        })
      );
    };

    return () => {
      source.close();
    };
  }, [runId]);

  const statusLabel = useMemo(() => {
    if (!runId) return t("status_idle");
    return kpi?.phase ? kpi.phase : t("status_running");
  }, [runId, kpi, locale]);
  const hasErrors = (kpi?.errors.count ?? 0) > 0;
  const verdict = useMemo(() => {
    if (!kpi) {
      return {
        label: t("verdict_idle"),
        detail: t("verdict_waiting"),
        tone: "",
      };
    }
    if (!realtime) {
      return {
        label: t("verdict_throughput"),
        detail: t("verdict_throughput_detail"),
        tone: "",
      };
    }
    const rtf = kpi.rtf?.p95;
    const errorRate = kpi.errors?.rate ?? 0;
    let tone = "";
    if (
      (typeof rtf === "number" && rtf > 1.2) ||
      (typeof errorRate === "number" && errorRate > 0.05)
    ) {
      tone = "error";
    } else if (
      (typeof rtf === "number" && rtf > 1.0) ||
      (typeof errorRate === "number" && errorRate > 0.01)
    ) {
      tone = "warn";
    }
    const label =
      tone === "error"
        ? t("verdict_bad")
        : tone === "warn"
          ? t("verdict_warn")
          : t("verdict_ok");
    const detail = `RTF p95 ${typeof rtf === "number" ? rtf.toFixed(3) : "--"} · Error ${
      (errorRate * 100).toFixed(1)
    }%`;
    return { label, detail, tone };
  }, [kpi, realtime, locale]);
  const showGpu = hasGpu && gpuSeries.length > 0;
  const displayStatus = runId ? resource : targetStatus;
  const runtimeInfo = runtimeSnapshot ?? displayStatus?.runtime ?? null;
  const runtimeAgeSec =
    runtimeTs != null ? Math.max(0, Date.now() / 1000 - runtimeTs) : null;
  const sessionHeader = sessionsPreview.header;
  const sessionSuccessIdx = sessionHeader.indexOf("success");
  const sessionErrorIdx = sessionHeader.indexOf("error_code");

  const startRun = async () => {
    if (!audioPath) {
      alert(t("alert_upload_first"));
      return;
    }
    if (!targetId) {
      alert(t("alert_select_target"));
      return;
    }
    const payload = {
      target_id: targetId,
      audio_path: audioPath,
      channels,
      duration_sec: durationSec,
      ramp_steps: rampSteps,
      ramp_interval_sec: rampInterval,
      chunk_ms: chunkMs,
      realtime,
      speed,
      task: "transcribe",
      language: language || null,
      decode_profile: decodeProfile,
      vad_mode: vadMode,
      attrs: parseKeyValueLines(attrsText),
      metadata: parseKeyValueLines(metadataText),
      token,
    };
    const res = await fetch(`${CONTROLLER_BASE}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      alert(`${t("alert_start_run_failed")}: ${detail}`);
      return;
    }
    const data = await res.json();
    setRunId(data.run_id);
    setLogs([]);
    setLatencySeries([]);
    setThroughputSeries([]);
    setGpuSeries([]);
    setHasGpu(false);
  };

  const stopRun = async () => {
    if (!runId) return;
    await fetch(`${CONTROLLER_BASE}/runs/${runId}/stop`, { method: "POST" });
  };

  const handleUpload = async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${CONTROLLER_BASE}/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      alert(t("alert_upload_failed"));
      return;
    }
    const data = await res.json();
    setAudioPath(data.audio_path);
  };

  const handleUploadBlob = async (blob: Blob, filename: string) => {
    const file = new File([blob], filename, { type: "audio/wav" });
    await handleUpload(file);
  };

  const startRecording = async () => {
    if (recording) return;
    setRecordStatus(t("record_requesting"));
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecordStatus(t("record_processing"));
        try {
          const blob = new Blob(recordChunksRef.current, {
            type: recorder.mimeType || "audio/webm",
          });
          const wavBlob = await convertBlobToWav(blob, 16000);
          const filename = `mic-${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
          setUploadName(filename);
          await handleUploadBlob(wavBlob, filename);
          setRecordStatus(t("record_upload_complete"));
        } catch (err) {
          console.error(err);
          setRecordStatus(t("record_failed"));
        } finally {
          setRecording(false);
          if (recordTimerRef.current) {
            window.clearInterval(recordTimerRef.current);
            recordTimerRef.current = null;
          }
          setRecordSecs(0);
        }
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      setRecordStatus(t("record_recording"));
      setRecordSecs(0);
      recordTimerRef.current = window.setInterval(() => {
        setRecordSecs((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error(err);
      setRecordStatus(t("record_permission_denied"));
      setRecording(false);
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;
    if (recorder.state !== "inactive") {
      recorder.stop();
    }
  };

  return (
    <main>
      <div className="dashboard">
        <div className="d-container">
          <section className="hero">
            <div className="hero-row">
              <span className={`status-pill ${runId ? "" : "warn"}`}>
                {statusLabel}
              </span>
              <div className="hero-actions">
                <LangToggle value={locale} onChange={setLocale} />
                <ThemeToggle />
              </div>
            </div>
            <h1>{t("title")}</h1>
            <p>{t("subtitle")}</p>
          </section>

          <div className="layout">
            <aside className="sidebar">
              <div className="card">
                <h3>
                  <span className="d-truncate">{t("run_control")}</span>
                </h3>
                <div className="form">
                  <label className="field">
                    {t("input_source")}
                    <select
                      value={inputMode}
                      onChange={(e) => setInputMode(e.target.value as "file" | "mic")}
                      disabled={recording}
                    >
                      <option value="file">{t("input_file")}</option>
                      <option value="mic">{t("input_mic")}</option>
                    </select>
                  </label>
                  <label className="field">
                    <div className="field-label">
                      {t("target")}
                      <HelpIcon text={t("help_target")} />
                    </div>
                    <select
                      value={targetId}
                      onChange={(e) => setTargetId(e.target.value)}
                    >
                      {targets.map((target) => (
                        <option key={target.id} value={target.id}>
                          {target.id} ({target.grpc_target})
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    {inputMode === "file" ? (
                      <>
                        {t("wav_upload")}
                        <div className="file-upload">
                          <button
                            type="button"
                            className="button secondary"
                            onClick={() => fileInputRef.current?.click()}
                          >
                            {t("choose_wav")}
                          </button>
                          <span
                            className={`file-name ${audioPath ? "ready" : "placeholder"}`}
                          >
                            {audioPath ? uploadName : t("no_file_selected")}
                          </span>
                        </div>
                        <input
                          ref={fileInputRef}
                          className="file-input"
                          type="file"
                          accept="audio/wav"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (!file) {
                              setUploadName("");
                              return;
                            }
                            setUploadName(file.name);
                            handleUpload(file);
                          }}
                        />
                      </>
                    ) : (
                      <>
                        {t("input_mic")}
                        <div className="mic-controls">
                          <button
                            type="button"
                            className={`button ${recording ? "danger" : "secondary"}`}
                            onClick={recording ? stopRecording : startRecording}
                          >
                            {recording ? t("stop") : t("record")}
                          </button>
                          <button
                            type="button"
                            className="button secondary"
                            onClick={stopRecording}
                            disabled={!recording}
                          >
                            {t("finish")}
                          </button>
                          <span className="mic-timer">{formatSeconds(recordSecs)}</span>
                        </div>
                        <div className="mic-status">{recordStatus}</div>
                        <div className="file-upload" style={{ marginTop: 10 }}>
                          <span
                            className={`file-name ${audioPath ? "ready" : "placeholder"}`}
                          >
                            {audioPath ? uploadName : t("no_recording_yet")}
                          </span>
                        </div>
                      </>
                    )}
                  </label>
                  <div className="pill-list">
                    <span className="pill">
                      {audioPath ? t("file_ready") : t("no_file")}
                    </span>
                    <span className="pill">{targetId || t("no_target")}</span>
                  </div>
                  <div className="row">
                    <label className="field">
                      {t("channels")}
                      <input
                        type="number"
                        min={1}
                        value={channels}
                        onChange={(e) => setChannels(Number(e.target.value))}
                      />
                    </label>
                    <label className="field">
                      {t("duration_sec")}
                      <input
                        type="number"
                        min={1}
                        value={durationSec}
                        onChange={(e) => setDurationSec(Number(e.target.value))}
                      />
                    </label>
                  </div>
                  <div className="row">
                    <label className="field">
                      <div className="field-label">
                        {t("ramp_steps")}
                        <HelpIcon text={t("help_ramp_steps")} />
                      </div>
                      <input
                        type="number"
                        min={1}
                        value={rampSteps}
                        onChange={(e) => setRampSteps(Number(e.target.value))}
                      />
                    </label>
                    <label className="field">
                      <div className="field-label">
                        {t("ramp_interval")}
                        <HelpIcon text={t("help_ramp_interval")} />
                      </div>
                      <input
                        type="number"
                        min={0}
                        step={0.5}
                        value={rampInterval}
                        onChange={(e) => setRampInterval(Number(e.target.value))}
                      />
                    </label>
                  </div>
                  <div className="row">
                    <label className="field">
                      <div className="field-label">
                        {t("chunk_ms")}
                        <HelpIcon text={t("help_chunk_ms")} />
                      </div>
                      <input
                        type="number"
                        min={10}
                        value={chunkMs}
                        onChange={(e) => setChunkMs(Number(e.target.value))}
                      />
                    </label>
                    <label className="field">
                      {t("speed")}
                      <input
                        type="number"
                        min={0.1}
                        step={0.1}
                        value={speed}
                        onChange={(e) => setSpeed(Number(e.target.value))}
                      />
                    </label>
                  </div>
                  <div className="row">
                    <label className="field">
                      {t("decode_profile")}
                      <select
                        value={decodeProfile}
                        onChange={(e) => setDecodeProfile(e.target.value)}
                      >
                        <option value="realtime">realtime</option>
                        <option value="accurate">accurate</option>
                      </select>
                    </label>
                    <label className="field">
                      <div className="field-label">
                        {t("vad_mode")}
                        <HelpIcon text={t("help_vad_mode")} />
                      </div>
                      <select
                        value={vadMode}
                        onChange={(e) => setVadMode(e.target.value)}
                      >
                        <option value="auto">auto</option>
                        <option value="continue">continue</option>
                      </select>
                    </label>
                  </div>
                  <div className="row">
                    <label className="field">
                      {t("language_optional")}
                      <select
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                      >
                        <option value="">{t("auto")}</option>
                        {languageOptions.map((option) => (
                          <option key={option.code} value={option.code}>
                            {option.name} ({option.code})
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      {t("auth_token_optional")}
                      <input
                        type="text"
                        placeholder="Bearer <ts>:<sig>"
                        value={token}
                        onChange={(e) => setToken(e.target.value)}
                      />
                    </label>
                  </div>
                  <label className="field">
                    <div className="field-label">
                      {t("attributes")}
                      <HelpIcon text={t("help_attrs_metadata")} />
                    </div>
                    <textarea
                      rows={3}
                      placeholder="partial=true"
                      value={attrsText}
                      onChange={(e) => setAttrsText(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    {t("metadata")}
                    <textarea
                      rows={3}
                      placeholder="authorization=Bearer <sig>\nx-stt-auth-ts=1700000000"
                      value={metadataText}
                      onChange={(e) => setMetadataText(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    <div className="field-label">
                      {t("realtime_pacing")}
                      <HelpIcon text={t("help_realtime_pacing")} />
                    </div>
                    <select
                      value={realtime ? "true" : "false"}
                      onChange={(e) => setRealtime(e.target.value === "true")}
                    >
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  </label>
                  <div className="actions">
                    <button className="button" onClick={startRun} disabled={!!runId}>
                      {t("start_run")}
                    </button>
                    <button
                      className="button danger"
                      onClick={stopRun}
                      disabled={!runId}
                    >
                      {t("stop_run")}
                    </button>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3>
                  <span className="d-truncate">{t("resource_status")}</span>
                </h3>
                {(() => {
                  const display = displayStatus;
                  const showGpuLine =
                    !!display &&
                    hasGpu &&
                    (display?.gpu_util_pct !== null || display?.vram_gb !== null);
                  return (
                    <div>
                      <div className="pill-list">
                        <span className={`pill ${display?.grpc_ok ? "" : "error"}`}>
                          gRPC {display?.grpc_ok ? "OK" : t("status_down")}
                        </span>
                        <span className={`pill ${display?.system_ok ? "" : "error"}`}>
                          /system {display?.system_ok ? "OK" : t("status_down")}
                        </span>
                        <span className={`pill ${display?.metrics_ok ? "" : "error"}`}>
                          /metrics {display?.metrics_ok ? "OK" : t("status_down")}
                        </span>
                      </div>
                      <div
                        style={{
                          marginTop: 12,
                          fontSize: 13,
                          color: "var(--d-text-muted)",
                        }}
                      >
                        CPU: {display?.cpu_pct ?? "--"}% · Mem: {display?.mem_gb ?? "--"} GB
                        {showGpuLine ? (
                          <>
                            <br />
                            GPU: {display?.gpu_util_pct ?? "--"}% · VRAM:{" "}
                            {display?.vram_gb ?? "--"} GB
                          </>
                        ) : null}
                      </div>
                    </div>
                  );
                })()}
              </div>

              <div className="card">
                <h3>
                  <span className="d-truncate">{t("server_config")}</span>
                </h3>
                {runtimeInfo ? (
                  (() => {
                    const model = runtimeInfo.model ?? {};
                    const streaming = runtimeInfo.streaming ?? {};
                    const modelBits = [
                      model.model_size,
                      model.model_backend,
                      model.device,
                      model.compute_type,
                    ].filter((value) => value);
                    const modelLine = modelBits.length ? modelBits.join(" · ") : "--";
                    return (
                      <div className="server-config">
                        <div>
                          <strong>Model</strong> {modelLine}
                        </div>
                        <div>
                          <strong>Model pool</strong>{" "}
                          {model.model_pool_size ?? "--"}
                        </div>
                        <div>
                          <strong>VAD pool</strong>{" "}
                          {streaming.vad_model_pool_size ?? "--"}{" "}
                          {streaming.vad_model_prewarm != null
                            ? `(prewarm ${streaming.vad_model_prewarm})`
                            : ""}
                        </div>
                        <div>
                          <strong>Default decode</strong>{" "}
                          {model.default_decode_profile ?? "--"}
                        </div>
                        <div>
                          <strong>Sample rate</strong>{" "}
                          {streaming.sample_rate ?? "--"} Hz
                        </div>
                        <div>
                          <strong>CreateSession</strong>{" "}
                          {streaming.create_session_rps ?? "--"} rps /{" "}
                          {streaming.create_session_burst ?? "--"} burst
                        </div>
                        <div>
                          <strong>Session timeout</strong>{" "}
                          {streaming.session_timeout_sec ?? "--"} s
                        </div>
                        <div>
                          <strong>VAD</strong>{" "}
                          {streaming.vad_threshold ?? "--"} /{" "}
                          {streaming.vad_silence ?? "--"} s
                        </div>
                        <div>
                          <strong>Chunk max</strong>{" "}
                          {streaming.max_chunk_ms ?? "--"} ms
                        </div>
                        <div>
                          <strong>Partial</strong>{" "}
                          {streaming.partial_decode_interval_sec ?? "--"} s /{" "}
                          {streaming.partial_decode_window_sec ?? "--"} s
                        </div>
                        <div>
                          <strong>Batch</strong>{" "}
                          {streaming.decode_batch_window_ms ?? "--"} ms /{" "}
                          {streaming.max_decode_batch_size ?? "--"}
                        </div>
                        {runtimeAgeSec != null ? (
                          <div className="server-config-meta">
                            {t("server_config_stale")} {Math.round(runtimeAgeSec)}s
                          </div>
                        ) : null}
                      </div>
                    );
                  })()
                ) : (
                  <div className="server-config">
                    {t("server_config_unavailable")}
                  </div>
                )}
              </div>
            </aside>

            <section className="panel">
              <div className="panel-header">
                <h2>{t("live_kpis")}</h2>
                <span className={`status-pill ${verdict.tone}`.trim()}>
                  {verdict.label}
                </span>
              </div>
              {verdict.detail ? (
                <div className="panel-sub">{verdict.detail}</div>
              ) : null}
              <div className="grid kpi-grid">
                <MetricCard
                  label="Active / Target"
                  value={`${kpi?.active ?? 0} / ${kpi?.target ?? 0}`}
                  className="highlight"
                />
                <MetricCard
                  label={t("latency_p95")}
                  value={((kpi?.latency_ms.p95 ?? 0) / 1000).toFixed(3)}
                  unit="s"
                  className="highlight"
                />
                <MetricCard
                  label={t("latency_p99")}
                  value={((kpi?.latency_ms.p99 ?? 0) / 1000).toFixed(3)}
                  unit="s"
                />
                <MetricCard label={t("rtf_p95")} value={(kpi?.rtf.p95 ?? 0).toFixed(3)} />
                <MetricCard
                  label={t("throughput")}
                  value={(kpi?.throughput_audio_sps ?? 0).toFixed(3)}
                  unit="audio-sec/s"
                />
                <MetricCard
                  label={t("errors")}
                  value={(kpi?.errors.per_sec ?? 0).toFixed(3)}
                  unit="/s"
                  className={hasErrors ? "danger" : ""}
                />
              </div>

              <div className="panel">
                <h2>{t("latency_throughput")}</h2>
                <div className="card chart-card" style={{ height: 260 }}>
                  {latencySeries.length === 0 ? (
                    <div className="chart-empty">
                      <div className="chart-empty-pill">{t("ready_to_start")}</div>
                    </div>
                  ) : null}
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={latencySeries}>
                      <CartesianGrid stroke="var(--d-grid-line)" />
                      <XAxis dataKey="ts" hide />
                      <YAxis tickFormatter={(value: number) => value.toFixed(3)} />
                      <Tooltip
                        formatter={(value: number) =>
                          typeof value === "number" ? value.toFixed(3) : value
                        }
                      />
                      <Legend />
                      <Line type="monotone" dataKey="p95" stroke="var(--d-line-1)" dot={false} />
                      <Line type="monotone" dataKey="p99" stroke="var(--d-line-2)" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="card chart-card" style={{ height: 260 }}>
                  {throughputSeries.length === 0 ? (
                    <div className="chart-empty">
                      <div className="chart-empty-pill">{t("ready_to_start")}</div>
                    </div>
                  ) : null}
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={throughputSeries}>
                      <CartesianGrid stroke="var(--d-grid-line)" />
                      <XAxis dataKey="ts" hide />
                      <YAxis />
                      <Tooltip
                        formatter={(value: number, name: string) => {
                          if (typeof value === "number") {
                            if (name === "active") return value.toFixed(0);
                            return value.toFixed(3);
                          }
                          return value;
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="active" stroke="var(--d-line-3)" dot={false} />
                      <Line type="monotone" dataKey="throughput" stroke="var(--d-line-1)" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {showGpu ? (
                <div className="panel">
                  <h2>{t("gpu_utilization")}</h2>
                  <div className="card chart-card" style={{ height: 260 }}>
                    {gpuSeries.length === 0 ? (
                      <div className="chart-empty">
                        <div className="chart-empty-pill">{t("ready_to_start")}</div>
                      </div>
                    ) : null}
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={gpuSeries}>
                        <CartesianGrid stroke="var(--d-grid-line)" />
                        <XAxis dataKey="ts" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="gpu"
                          stroke="var(--d-line-2)"
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="vram"
                          stroke="var(--d-line-1)"
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : null}

              <div className="panel">
                <h2>{t("run_logs")}</h2>
                <div className="log-box">
                  {logs.length === 0
                    ? t("logs_empty")
                    : logs.map((entry, idx) => (
                        <div
                          key={`${entry.ts}-${idx}`}
                          className={`log-line ${entry.level.toLowerCase()}`}
                        >
                          [{new Date(entry.ts * 1000).toLocaleTimeString()}] {entry.level}
                          {" "}
                          {entry.source}: {entry.msg}
                        </div>
                      ))}
                </div>
              </div>

              <div className="panel">
                <div className="panel-header">
                  <h2>{t("sessions_preview")}</h2>
                  {sessionsPreview.run_id ? (
                    <span
                      className="panel-sub"
                      title={sessionsPreview.run_id}
                    >
                      {sessionsPreview.run_id.slice(0, 8)}
                    </span>
                  ) : null}
                </div>
                <div className="table-scroll">
                  {sessionsPreview.header.length === 0 ||
                  sessionsPreview.rows.length === 0 ? (
                    <div className="table-empty">
                      {t("sessions_preview_empty")}
                    </div>
                  ) : (
                    <table className="sessions-table">
                      <thead>
                        <tr>
                          {sessionsPreview.header.map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sessionsPreview.rows.map((row, idx) => {
                          const successValue =
                            sessionSuccessIdx >= 0
                              ? row[sessionSuccessIdx]
                              : "";
                          const errorValue =
                            sessionErrorIdx >= 0 ? row[sessionErrorIdx] : "";
                          const isError =
                            (successValue &&
                              successValue.toLowerCase() !== "true") ||
                            (errorValue && errorValue !== "None");
                          return (
                            <tr
                              key={`${idx}-${row[0] ?? "row"}`}
                              className={isError ? "error" : ""}
                            >
                              {sessionHeader.map((_, colIdx) => (
                                <td key={`${idx}-${colIdx}`}>
                                  {row[colIdx] ?? ""}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              <div className="panel">
                <h2>{t("run_history_compare")}</h2>
                <div className="card">
                  <div className="row">
                    <label className="field">
                      {t("compare_a")}
                      <select
                        value={compareA}
                        onChange={(e) => setCompareA(e.target.value)}
                      >
                        <option value="">{t("select_run")}</option>
                        {history.map((run) => (
                          <option key={run.run_id} value={run.run_id}>
                            {run.run_id}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="field">
                      {t("compare_b")}
                      <select
                        value={compareB}
                        onChange={(e) => setCompareB(e.target.value)}
                      >
                        <option value="">{t("select_run")}</option>
                        {history.map((run) => (
                          <option key={run.run_id} value={run.run_id}>
                            {run.run_id}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="grid" style={{ marginTop: 16 }}>
                    <SummaryCard title="Run A" run={findRun(history, compareA)} t={t} />
                    <SummaryCard title="Run B" run={findRun(history, compareB)} t={t} />
                  </div>
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
  unit,
  className,
}: {
  label: string;
  value: string | number;
  unit?: string;
  className?: string;
}) {
  return (
    <div className={`card ${className ?? ""}`.trim()}>
      <h3>
        <span className="d-truncate">{label}</span>
      </h3>
      <div className="metric">
        <span className="metric-value">{value}</span>
        {unit ? <span className="metric-unit">{unit}</span> : null}
      </div>
    </div>
  );
}

function HelpIcon({ text }: { text: string }) {
  return (
    <span className="help-icon" tabIndex={0} role="button" aria-label={text}>
      ?
      <span className="help-tooltip" role="tooltip">
        {text}
      </span>
    </span>
  );
}

function appendSeries(prev: any[], next: any, max = 120) {
  const updated = [...prev, next];
  if (updated.length > max) {
    return updated.slice(updated.length - max);
  }
  return updated;
}

function appendLogs(prev: LogEntry[], next: LogEntry, max = 200) {
  const updated = [...prev, next];
  if (updated.length > max) {
    return updated.slice(updated.length - max);
  }
  return updated;
}

function parseKeyValueLines(raw: string): Record<string, string> {
  if (!raw) return {};
  const result: Record<string, string> = {};
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .forEach((line) => {
      const idx = line.indexOf("=");
      if (idx === -1) return;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      if (key) result[key] = value;
    });
  return result;
}

function findRun(history: any[], runId: string) {
  if (!runId) return null;
  return history.find((run) => run.run_id === runId) ?? null;
}

function SummaryCard({
  title,
  run,
  t,
}: {
  title: string;
  run: any;
  t: (key: string) => string;
}) {
  if (!run) {
    return (
      <div className="card">
        <h3>
          <span className="d-truncate">{title}</span>
        </h3>
        <div className="metric">
          <span className="metric-value">--</span>
        </div>
      </div>
    );
  }
  const summary = run.summary || {};
  const latency = summary.latency_ms || {};
  const rtf = summary.rtf || {};
  const p95 =
    typeof latency.p95 === "number" ? (latency.p95 / 1000).toFixed(3) : "--";
  const p99 =
    typeof latency.p99 === "number" ? (latency.p99 / 1000).toFixed(3) : "--";
  const rtf95 = typeof rtf.p95 === "number" ? rtf.p95.toFixed(3) : "--";
  const throughput =
    typeof summary.throughput_audio_sps === "number"
      ? summary.throughput_audio_sps.toFixed(3)
      : "--";
  return (
    <div className="card">
      <h3>
        <span className="d-truncate">{title}</span>
      </h3>
      <div className="metric">
        <span className="metric-value">{p95}</span>
        <span className="metric-unit">p95 s</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 13, color: "var(--muted)" }}>
        p99: {p99} s<br />
        RTF p95: {rtf95}
        <br />
        Throughput: {throughput} audio-sec/s
      </div>
      <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <a
          className="button secondary"
          href={`${CONTROLLER_BASE}/runs/${run.run_id}/report`}
          target="_blank"
          rel="noreferrer"
        >
          {t("download_report")}
        </a>
        <a
          className="button secondary"
          href={`${CONTROLLER_BASE}/runs/${run.run_id}/sessions.csv`}
          target="_blank"
          rel="noreferrer"
        >
          {t("download_sessions")}
        </a>
      </div>
    </div>
  );
}

async function convertBlobToWav(blob: Blob, targetRate: number) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  const mixed = mixDownToMono(decoded);
  const rendered =
    decoded.sampleRate === targetRate
      ? mixed
      : await resampleBuffer(mixed, decoded.sampleRate, targetRate);
  return encodeWav(rendered, targetRate);
}

function mixDownToMono(buffer: AudioBuffer): Float32Array {
  const channels = buffer.numberOfChannels;
  if (channels === 1) {
    return buffer.getChannelData(0);
  }
  const length = buffer.length;
  const mix = new Float32Array(length);
  for (let ch = 0; ch < channels; ch += 1) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < length; i += 1) {
      mix[i] += data[i];
    }
  }
  for (let i = 0; i < length; i += 1) {
    mix[i] /= channels;
  }
  return mix;
}

async function resampleBuffer(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number
) {
  const duration = samples.length / sourceRate;
  const frameCount = Math.ceil(duration * targetRate);
  const offline = new OfflineAudioContext(1, frameCount, targetRate);
  const buffer = offline.createBuffer(1, samples.length, sourceRate);
  buffer.getChannelData(0).set(samples);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

function encodeWav(samples: Float32Array, sampleRate: number) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    let s = samples[i];
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, value: string) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function formatSeconds(total: number) {
  const minutes = Math.floor(total / 60)
    .toString()
    .padStart(2, "0");
  const seconds = Math.floor(total % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}
