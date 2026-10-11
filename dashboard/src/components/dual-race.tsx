"use client";

import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const DUAL_WS =
  process.env.NEXT_PUBLIC_DUAL_WS_URL ?? "ws://127.0.0.1:8000/ws/dual";

type DualState = {
  running: boolean;
  status: string;
  message: string;
  episode: number;
  total_episodes: number;
  bc_score: number;
  dqn_score: number;
  bc_alive: boolean;
  dqn_alive: boolean;
  bc_total: number;
  dqn_total: number;
  bc_frame_b64: string;
  dqn_frame_b64: string;
  winner: string;
  episode_winners: string[];
};

const empty: DualState = {
  running: false,
  status: "idle",
  message: "Start a dual race to watch BC vs DQN play side by side",
  episode: 0,
  total_episodes: 0,
  bc_score: 0,
  dqn_score: 0,
  bc_alive: false,
  dqn_alive: false,
  bc_total: 0,
  dqn_total: 0,
  bc_frame_b64: "",
  dqn_frame_b64: "",
  winner: "",
  episode_winners: [],
};

export function DualRacePanel({ apiOnline }: { apiOnline: boolean }) {
  const [state, setState] = useState<DualState>(empty);
  const [episodes, setEpisodes] = useState(5);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!apiOnline) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = new WebSocket(DUAL_WS);
      ws.onmessage = (ev) => {
        try {
          setState(JSON.parse(ev.data) as DualState);
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [apiOnline]);

  const start = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/api/dual/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ episodes }),
      });
    } finally {
      setBusy(false);
    }
  }, [episodes]);

  const stop = useCallback(async () => {
    setBusy(true);
    try {
      await fetch(`${API_BASE}/api/dual/stop`, { method: "POST" });
    } finally {
      setBusy(false);
    }
  }, []);

  const winnerLabel =
    state.winner === "tie"
      ? "Tie"
      : state.winner
        ? `${state.winner} wins!`
        : "—";

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle>Dual race — two Flappy Birds</CardTitle>
          <CardDescription>
            Watch BC-CNN and DQN-CNN play at the same time from trained
            checkpoints, then see who wins on total pipes passed.
          </CardDescription>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">
            Ep {state.episode}/{state.total_episodes || episodes}
          </Badge>
          {state.winner && <Badge>{winnerLabel}</Badge>}
          <Badge variant={state.running ? "default" : "outline"}>
            {state.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            Race episodes
            <input
              type="number"
              min={1}
              max={30}
              value={episodes}
              onChange={(e) => setEpisodes(Number(e.target.value) || 5)}
              className="h-8 w-20 rounded-md border bg-background px-2 font-mono text-sm"
            />
          </label>
          <Button
            onClick={start}
            disabled={!apiOnline || busy || state.running}
          >
            Start dual race
          </Button>
          <Button
            variant="outline"
            onClick={stop}
            disabled={!apiOnline || busy || !state.running}
          >
            Stop
          </Button>
        </div>

        <p className="text-sm text-muted-foreground">{state.message}</p>

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium text-orange-600">BC-CNN</span>
              <span className="font-mono text-xs">
                score {state.bc_score} · total {state.bc_total}{" "}
                {state.bc_alive ? "· alive" : "· dead"}
              </span>
            </div>
            <div className="aspect-[800/576] overflow-hidden rounded-xl bg-zinc-950 ring-1 ring-foreground/10">
              {state.bc_frame_b64 ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:image/jpeg;base64,${state.bc_frame_b64}`}
                  alt="BC-CNN play"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full min-h-48 items-center justify-center text-xs text-zinc-500">
                  Waiting for BC frames…
                </div>
              )}
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium text-orange-700">DQN-CNN</span>
              <span className="font-mono text-xs">
                score {state.dqn_score} · total {state.dqn_total}{" "}
                {state.dqn_alive ? "· alive" : "· dead"}
              </span>
            </div>
            <div className="aspect-[800/576] overflow-hidden rounded-xl bg-zinc-950 ring-1 ring-foreground/10">
              {state.dqn_frame_b64 ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:image/jpeg;base64,${state.dqn_frame_b64}`}
                  alt="DQN-CNN play"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full min-h-48 items-center justify-center text-xs text-zinc-500">
                  Waiting for DQN frames…
                </div>
              )}
            </div>
          </div>
        </div>

        {state.episode_winners.length > 0 && (
          <div className="text-xs text-muted-foreground">
            Episode winners: {state.episode_winners.join(" → ")}
          </div>
        )}

        {!state.running && state.winner && (
          <div className="rounded-lg border bg-muted/40 px-4 py-3 text-sm font-medium">
            Final result: <span className="text-primary">{winnerLabel}</span>
            {" — "}
            BC total pipes {state.bc_total} vs DQN total pipes {state.dqn_total}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
