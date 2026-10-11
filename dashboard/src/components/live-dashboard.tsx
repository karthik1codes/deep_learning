"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DualRacePanel } from "@/components/dual-race";
import { ModelComparisonPanel } from "@/components/model-comparison";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws/live";

type LiveState = {
  running: boolean;
  status: string;
  step: number;
  episode: number;
  epsilon: number;
  last_loss: number;
  episode_reward: number;
  last_score: number;
  mean_score_10: number;
  max_score: number;
  fps: number;
  action: number;
  message: string;
  scores: number[];
  losses: number[];
  rewards: number[];
  frame_jpeg_b64: string;
  updated_at: number;
};

const emptyState: LiveState = {
  running: false,
  status: "idle",
  step: 0,
  episode: 0,
  epsilon: 1,
  last_loss: 0,
  episode_reward: 0,
  last_score: 0,
  mean_score_10: 0,
  max_score: 0,
  fps: 0,
  action: 0,
  message: "Connect API then press Start Training",
  scores: [],
  losses: [],
  rewards: [],
  frame_jpeg_b64: "",
  updated_at: 0,
};

function Sparkline({
  values,
  stroke = "currentColor",
}: {
  values: number[];
  stroke?: string;
}) {
  const points = useMemo(() => {
    if (!values.length) return "";
    const slice = values.slice(-80);
    const min = Math.min(...slice);
    const max = Math.max(...slice);
    const span = max - min || 1;
    return slice
      .map((v, i) => {
        const x = (i / Math.max(slice.length - 1, 1)) * 100;
        const y = 100 - ((v - min) / span) * 100;
        return `${x},${y}`;
      })
      .join(" ");
  }, [values]);

  return (
    <svg viewBox="0 0 100 100" className="h-24 w-full" preserveAspectRatio="none">
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="2"
        points={points || "0,50 100,50"}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border bg-muted/40 px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-sm font-medium tabular-nums">{value}</div>
    </div>
  );
}

export function LiveDashboard() {
  const [state, setState] = useState<LiveState>(emptyState);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [timesteps, setTimesteps] = useState(15000);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        if (!closed) setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws?.close();
      ws.onmessage = (ev) => {
        try {
          setState(JSON.parse(ev.data) as LiveState);
        } catch {
          /* ignore bad frames */
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, []);

  const start = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/api/train/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          timesteps,
          warmup: 500,
          init_bc: "cnn_flappy/checkpoints/bc_cnn.pt",
          frame_every: 2,
        }),
      });
    } finally {
      setBusy(false);
    }
  }, [timesteps]);

  const stop = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/api/train/stop`, { method: "POST" });
    } finally {
      setBusy(false);
    }
  }, []);

  const frameSrc = state.frame_jpeg_b64
    ? `data:image/jpeg;base64,${state.frame_jpeg_b64}`
    : null;

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 p-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-heading text-3xl font-semibold tracking-tight">
            CNN Flappy Live
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Live gameplay + DQN training metrics, plus BC-CNN vs DQN-CNN
            performance comparison for result analysis. Next.js + shadcn/ui.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={connected ? "default" : "destructive"}>
            {connected ? "API connected" : "API offline"}
          </Badge>
          <Badge variant="secondary">{state.status}</Badge>
        </div>
      </header>

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle>Training controls</CardTitle>
            <CardDescription>{state.message}</CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              Steps
              <input
                type="number"
                min={1000}
                step={1000}
                value={timesteps}
                onChange={(e) => setTimesteps(Number(e.target.value) || 15000)}
                className="h-8 w-28 rounded-md border bg-background px-2 font-mono text-sm"
              />
            </label>
            <Button onClick={start} disabled={busy || state.running || !connected}>
              Start Training
            </Button>
            <Button
              variant="outline"
              onClick={stop}
              disabled={busy || !state.running}
            >
              Stop
            </Button>
          </div>
        </CardHeader>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle>Gameplay</CardTitle>
            <CardDescription>
              Live RGB frames from the Flappy Bird env (pygame surface → JPEG
              stream)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-[800/576] overflow-hidden rounded-xl bg-zinc-950 ring-1 ring-foreground/10">
              {frameSrc ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={frameSrc}
                  alt="Flappy Bird live frame"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full min-h-64 items-center justify-center text-sm text-zinc-400">
                  Waiting for frames… start training on a connected API.
                </div>
              )}
              <div className="absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 font-mono text-xs text-white">
                action: {state.action === 1 ? "FLAP" : "IDLE"}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live training metrics</CardTitle>
            <CardDescription>
              CNN-DQN self-improvement signals updating in real time
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <Metric label="Step" value={state.step} />
              <Metric label="Episode" value={state.episode} />
              <Metric label="Epsilon" value={state.epsilon} />
              <Metric label="TD loss" value={state.last_loss} />
              <Metric label="Ep reward" value={state.episode_reward} />
              <Metric label="Last score" value={state.last_score} />
              <Metric label="Mean10" value={state.mean_score_10} />
              <Metric label="Max score" value={state.max_score} />
              <Metric label="Train FPS" value={state.fps} />
            </div>

            <Separator />

            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">
                Episode scores
              </div>
              <div className="rounded-lg border bg-muted/30 p-2 text-chart-1">
                <Sparkline values={state.scores} stroke="oklch(0.55 0.15 250)" />
              </div>
            </div>

            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground">
                TD loss
              </div>
              <div className="rounded-lg border bg-muted/30 p-2">
                <Sparkline values={state.losses} stroke="oklch(0.65 0.16 50)" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <DualRacePanel apiOnline={connected} />
      <ModelComparisonPanel apiOnline={connected} />
    </div>
  );
}
