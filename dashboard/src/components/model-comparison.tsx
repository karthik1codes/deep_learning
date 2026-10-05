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
import { Separator } from "@/components/ui/separator";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type AgentResult = {
  name: string;
  scores: number[];
  rewards: number[];
  mean_score: number;
  max_score: number;
  mean_reward: number;
  episodes: number;
};

type CompareResult = {
  episodes: number;
  winner: string | null;
  skipped: string[];
  results: AgentResult[];
  chart_png_b64: string;
  bc_vs_dqn: {
    bc_mean_score: number | null;
    dqn_mean_score: number | null;
    bc_max_score: number | null;
    dqn_max_score: number | null;
    bc_mean_reward: number | null;
    dqn_mean_reward: number | null;
  };
  models: {
    bc: { path: string; exists: boolean };
    dqn: { path: string; exists: boolean };
  };
};

type CompareResponse = {
  running: boolean;
  error: string | null;
  result: CompareResult | null;
};

function BarChart({ results }: { results: AgentResult[] }) {
  const max = Math.max(...results.map((r) => r.mean_score), 0.01);
  const colors: Record<string, string> = {
    random: "bg-sky-300",
    heuristic: "bg-emerald-300",
    "BC-CNN": "bg-orange-300",
    "DQN-CNN": "bg-orange-600",
  };

  return (
    <div className="space-y-3">
      {results.map((r) => (
        <div key={r.name} className="grid grid-cols-[100px_1fr_56px] items-center gap-2">
          <div className="truncate text-xs font-medium">{r.name}</div>
          <div className="h-3 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full ${colors[r.name] ?? "bg-primary"}`}
              style={{ width: `${(r.mean_score / max) * 100}%` }}
            />
          </div>
          <div className="text-right font-mono text-xs tabular-nums">
            {r.mean_score.toFixed(2)}
          </div>
        </div>
      ))}
    </div>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-muted/40 px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-sm font-medium">{value}</div>
    </div>
  );
}

export function ModelComparisonPanel({ apiOnline }: { apiOnline: boolean }) {
  const [episodes, setEpisodes] = useState(10);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [models, setModels] = useState<{
    bc: boolean;
    dqn: boolean;
  }>({ bc: false, dqn: false });

  const refreshModels = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
      if (!res.ok) return;
      const data = await res.json();
      setModels({ bc: !!data.bc?.exists, dqn: !!data.dqn?.exists });
    } catch {
      /* ignore */
    }
  }, []);

  const pollCompare = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/compare`);
      if (!res.ok) return;
      const data = (await res.json()) as CompareResponse;
      setRunning(data.running);
      setError(data.error);
      if (data.result) setResult(data.result);
      if (!data.running) setBusy(false);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!apiOnline) return;
    refreshModels();
    pollCompare();
  }, [apiOnline, refreshModels, pollCompare]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(pollCompare, 1500);
    return () => clearInterval(id);
  }, [running, pollCompare]);

  const startCompare = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          episodes,
          include_baselines: true,
        }),
      });
      const data = await res.json();
      if (!data.started) {
        setError(data.message ?? "Could not start comparison");
        setBusy(false);
        return;
      }
      setRunning(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Compare request failed");
      setBusy(false);
    }
  }, [episodes]);

  const winnerLabel =
    result?.winner === "tie"
      ? "Tie"
      : result?.winner
        ? `${result.winner} leads`
        : "—";

  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle>Model comparison — BC-CNN vs DQN-CNN</CardTitle>
          <CardDescription>
            AAT-ready analytics: evaluate both trained models on the same
            episodes and compare mean score, max score, and reward.
          </CardDescription>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={models.bc ? "default" : "outline"}>
            BC {models.bc ? "ready" : "missing"}
          </Badge>
          <Badge variant={models.dqn ? "default" : "outline"}>
            DQN {models.dqn ? "ready" : "missing"}
          </Badge>
          {result?.winner && (
            <Badge variant="secondary">{winnerLabel}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            Eval episodes
            <input
              type="number"
              min={2}
              max={50}
              value={episodes}
              onChange={(e) => setEpisodes(Number(e.target.value) || 10)}
              className="h-8 w-20 rounded-md border bg-background px-2 font-mono text-sm"
            />
          </label>
          <Button
            onClick={startCompare}
            disabled={!apiOnline || busy || running || (!models.bc && !models.dqn)}
          >
            {running ? "Comparing…" : "Run BC vs DQN comparison"}
          </Button>
          <Button variant="outline" onClick={refreshModels} disabled={!apiOnline}>
            Refresh models
          </Button>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
        {running && (
          <p className="text-sm text-muted-foreground">
            Evaluating agents in the background — this can take a minute…
          </p>
        )}

        {result && (
          <>
            <Separator />
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-3">
                <div className="text-sm font-medium">Performance table</div>
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-muted/50 text-xs text-muted-foreground">
                      <tr>
                        <th className="px-3 py-2">Model</th>
                        <th className="px-3 py-2">Mean score</th>
                        <th className="px-3 py-2">Max score</th>
                        <th className="px-3 py-2">Mean reward</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.results.map((r) => (
                        <tr key={r.name} className="border-t">
                          <td className="px-3 py-2 font-medium">{r.name}</td>
                          <td className="px-3 py-2 font-mono tabular-nums">
                            {r.mean_score.toFixed(2)}
                          </td>
                          <td className="px-3 py-2 font-mono tabular-nums">
                            {r.max_score}
                          </td>
                          <td className="px-3 py-2 font-mono tabular-nums">
                            {r.mean_reward.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  <MetricCell
                    label="BC mean score"
                    value={
                      result.bc_vs_dqn.bc_mean_score?.toFixed(2) ?? "n/a"
                    }
                  />
                  <MetricCell
                    label="DQN mean score"
                    value={
                      result.bc_vs_dqn.dqn_mean_score?.toFixed(2) ?? "n/a"
                    }
                  />
                  <MetricCell label="Winner" value={winnerLabel} />
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-sm font-medium">Mean score chart</div>
                <BarChart results={result.results} />
                {result.chart_png_b64 && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/png;base64,${result.chart_png_b64}`}
                    alt="BC vs DQN comparison chart"
                    className="mt-2 w-full rounded-lg border"
                  />
                )}
                {result.skipped.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Skipped: {result.skipped.join("; ")}
                  </p>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
