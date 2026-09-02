"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  Database,
  History,
  RefreshCw,
  ShieldCheck,
  Target,
  Trophy,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Fixture = {
  fixture_key: string;
  kickoff_utc: string;
  status: string;
  matchday: number | null;
  home_team: string;
  away_team: string;
  home_win: number;
  draw: number;
  away_win: number;
  expected_home_goals: number;
  expected_away_goals: number;
  predicted_outcome: "home_win" | "draw" | "away_win";
  predicted_score: string;
  prediction_created_at: string;
};

type Prediction = Fixture & {
  home_goals: number | null;
  away_goals: number | null;
  actual_outcome: "home_win" | "draw" | "away_win" | null;
  correct: number | null;
  log_loss: number | null;
  brier_score: number | null;
};

type RecordSummary = {
  predictions: number;
  graded: number;
  accuracy: number | null;
  log_loss: number | null;
  brier_score: number | null;
};

const demoFixtures: Fixture[] = [
  {
    fixture_key: "demo:chelsea-arsenal",
    kickoff_utc: "2026-09-12T14:00:00Z",
    status: "TIMED",
    matchday: 4,
    home_team: "Chelsea",
    away_team: "Arsenal",
    home_win: 0.234,
    draw: 0.214,
    away_win: 0.552,
    expected_home_goals: 1.15,
    expected_away_goals: 1.85,
    predicted_outcome: "away_win",
    predicted_score: "1-1",
    prediction_created_at: "2026-09-02T03:00:00Z",
  },
  {
    fixture_key: "demo:liverpool-everton",
    kickoff_utc: "2026-09-12T16:30:00Z",
    status: "TIMED",
    matchday: 4,
    home_team: "Liverpool",
    away_team: "Everton",
    home_win: 0.641,
    draw: 0.211,
    away_win: 0.148,
    expected_home_goals: 2.08,
    expected_away_goals: 0.82,
    predicted_outcome: "home_win",
    predicted_score: "2-0",
    prediction_created_at: "2026-09-02T03:00:00Z",
  },
  {
    fixture_key: "demo:city-newcastle",
    kickoff_utc: "2026-09-13T13:00:00Z",
    status: "TIMED",
    matchday: 4,
    home_team: "Man City",
    away_team: "Newcastle",
    home_win: 0.537,
    draw: 0.238,
    away_win: 0.225,
    expected_home_goals: 1.76,
    expected_away_goals: 1.11,
    predicted_outcome: "home_win",
    predicted_score: "1-1",
    prediction_created_at: "2026-09-02T03:00:00Z",
  },
  {
    fixture_key: "demo:spurs-villa",
    kickoff_utc: "2026-09-13T15:30:00Z",
    status: "TIMED",
    matchday: 4,
    home_team: "Tottenham",
    away_team: "Aston Villa",
    home_win: 0.418,
    draw: 0.271,
    away_win: 0.311,
    expected_home_goals: 1.49,
    expected_away_goals: 1.32,
    predicted_outcome: "home_win",
    predicted_score: "1-1",
    prediction_created_at: "2026-09-02T03:00:00Z",
  },
];

const demoHistory: Prediction[] = [
  {
    ...demoFixtures[1],
    fixture_key: "demo:history-1",
    kickoff_utc: "2026-08-31T15:30:00Z",
    home_team: "Aston Villa",
    away_team: "Arsenal",
    predicted_outcome: "away_win",
    home_win: 0.214,
    draw: 0.247,
    away_win: 0.539,
    predicted_score: "1-2",
    home_goals: 0,
    away_goals: 1,
    actual_outcome: "away_win",
    correct: 1,
    log_loss: 0.618,
    brier_score: 0.321,
  },
  {
    ...demoFixtures[0],
    fixture_key: "demo:history-2",
    kickoff_utc: "2026-08-30T14:00:00Z",
    home_team: "Chelsea",
    away_team: "Brighton",
    predicted_outcome: "home_win",
    home_win: 0.512,
    draw: 0.246,
    away_win: 0.242,
    predicted_score: "2-1",
    home_goals: 4,
    away_goals: 3,
    actual_outcome: "home_win",
    correct: 1,
    log_loss: 0.669,
    brier_score: 0.359,
  },
  {
    ...demoFixtures[2],
    fixture_key: "demo:history-3",
    kickoff_utc: "2026-08-29T16:30:00Z",
    home_team: "Liverpool",
    away_team: "Man City",
    predicted_outcome: "home_win",
    home_win: 0.421,
    draw: 0.282,
    away_win: 0.297,
    predicted_score: "1-1",
    home_goals: 1,
    away_goals: 1,
    actual_outcome: "draw",
    correct: 0,
    log_loss: 1.266,
    brier_score: 0.779,
  },
];

const demoRecord: RecordSummary = {
  predictions: 30,
  graded: 20,
  accuracy: 0.55,
  log_loss: 1.018,
  brier_score: 0.612,
};

function percentage(value: number | null) {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function metric(value: number | null, digits = 3) {
  return value === null ? "—" : value.toFixed(digits);
}

function fixtureDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function outcomeLabel(fixture: Fixture) {
  if (fixture.predicted_outcome === "home_win") return fixture.home_team;
  if (fixture.predicted_outcome === "away_win") return fixture.away_team;
  return "Draw";
}

function ResultMark({ correct }: { correct: number | null }) {
  if (correct === null) return <span className="text-slate-500">Pending</span>;
  return correct === 1 ? (
    <span className="inline-flex items-center gap-1.5 font-semibold text-[#b8ff3d]">
      <CheckCircle2 className="size-4" /> Correct
    </span>
  ) : (
    <span className="font-semibold text-rose-400">Missed</span>
  );
}

function ProbabilityRow({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-4 text-sm">
        <span className="truncate text-slate-300">{label}</span>
        <span className="font-mono text-base font-bold text-white">{percentage(value)}</span>
      </div>
      <Progress
        aria-label={`${label} probability ${percentage(value)}`}
        value={value * 100}
        className={`h-2.5 bg-white/8 ${tone}`}
      />
    </div>
  );
}

export default function Home() {
  const [fixtures, setFixtures] = useState<Fixture[]>(demoFixtures);
  const [history, setHistory] = useState<Prediction[]>(demoHistory);
  const [record, setRecord] = useState<RecordSummary>(demoRecord);
  const [selectedKey, setSelectedKey] = useState(demoFixtures[0].fixture_key);
  const [source, setSource] = useState<"loading" | "live" | "error">("loading");
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const response = await fetch("/api/dashboard", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("API unavailable");
      }
      const payload = (await response.json()) as {
        fixtures: Fixture[];
        predictions: Prediction[];
        record: RecordSummary;
      };
      setFixtures(payload.fixtures);
      setHistory(payload.predictions);
      setRecord(payload.record);
      setSelectedKey((current) =>
        payload.fixtures.some((fixture) => fixture.fixture_key === current)
          ? current
          : (payload.fixtures[0]?.fixture_key ?? ""),
      );
      setSource("live");
    } catch {
      setSource("error");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timeout);
  }, [load]);

  const selected = useMemo(
    () => fixtures.find((fixture) => fixture.fixture_key === selectedKey) ?? fixtures[0],
    [fixtures, selectedKey],
  );
  const gradedAccuracy = record.accuracy ?? 0;
  const highConfidence = fixtures.filter(
    (fixture) => Math.max(fixture.home_win, fixture.draw, fixture.away_win) >= 0.6,
  ).length;

  return (
    <main className="min-h-screen bg-[#070b12] text-slate-100">
      <div className="mx-auto min-h-screen max-w-[1480px] border-x border-white/7 bg-[#090e17]">
        <header className="flex flex-col gap-5 border-b border-white/8 px-5 py-5 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="grid size-11 place-items-center rounded-xl border border-[#b8ff3d]/30 bg-[#b8ff3d]/10 text-[#b8ff3d]">
              <Activity className="size-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-black tracking-[-0.03em] text-white">PL PREDICTOR</h1>
                <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs font-semibold text-slate-400">V8</span>
              </div>
              <p className="mt-1 text-sm text-slate-500">Premier League · Gameweek model</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-lg border border-white/8 bg-white/[0.035] px-3 py-2 text-sm">
              <span className={`size-2 rounded-full ${source === "live" ? "bg-[#b8ff3d] shadow-[0_0_12px_#b8ff3d]" : source === "loading" ? "animate-pulse bg-amber-300" : "bg-rose-400"}`} />
              <span className="text-slate-300">{source === "live" ? "Live API" : source === "loading" ? "Connecting" : "API unavailable"}</span>
            </div>
            <Button variant="outline" onClick={() => void load()} disabled={refreshing} className="border-white/10 bg-white/[0.035] text-slate-200 hover:bg-white/10 hover:text-white">
              <RefreshCw className={refreshing ? "animate-spin" : ""} /> Refresh
            </Button>
          </div>
        </header>

        <section className="grid border-b border-white/8 sm:grid-cols-2 xl:grid-cols-4">
          {[
            { label: "Upcoming", value: fixtures.length, detail: "tracked fixtures", icon: CalendarDays },
            { label: "Predictions", value: record.predictions, detail: `${record.graded} graded`, icon: Target },
            { label: "Live accuracy", value: percentage(record.accuracy), detail: "hard outcome", icon: Trophy },
            { label: "High confidence", value: highConfidence, detail: "60% or higher", icon: ShieldCheck },
          ].map((item, index) => (
            <article key={item.label} className={`flex items-center justify-between px-5 py-5 sm:px-8 ${index < 3 ? "xl:border-r xl:border-white/8" : ""}`}>
              <div><p className="text-sm text-slate-500">{item.label}</p><p className="mt-1 text-2xl font-black tracking-tight text-white">{item.value}</p><p className="mt-1 text-xs text-slate-600">{item.detail}</p></div>
              <item.icon className="size-5 text-[#b8ff3d]" />
            </article>
          ))}
        </section>

        <Tabs defaultValue="gameweek" className="gap-0">
          <div className="border-b border-white/8 px-5 sm:px-8">
            <TabsList variant="line" className="h-14 gap-6 text-slate-500">
              <TabsTrigger value="gameweek" className="px-0 text-sm data-[state=active]:text-white"><CalendarDays /> Gameweek</TabsTrigger>
              <TabsTrigger value="history" className="px-0 text-sm data-[state=active]:text-white"><History /> History</TabsTrigger>
              <TabsTrigger value="model" className="px-0 text-sm data-[state=active]:text-white"><BarChart3 /> Model</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="gameweek" className="m-0">
            <div className="grid min-h-[650px] xl:grid-cols-[0.85fr_1.15fr]">
              <section className="border-b border-white/8 px-5 py-7 sm:px-8 xl:border-r xl:border-b-0">
                <div className="mb-5 flex items-end justify-between gap-4">
                  <div><p className="text-xs font-bold uppercase tracking-[0.2em] text-[#b8ff3d]">Next fixtures</p><h2 className="mt-2 text-2xl font-black tracking-tight text-white">Gameweek board</h2></div>
                  <span className="font-mono text-xs text-slate-600">UTC</span>
                </div>
                {fixtures.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-white/12 p-8 text-center text-slate-500">No upcoming fixtures are stored yet.</div>
                ) : (
                  <div className="space-y-3">
                    {fixtures.map((fixture) => {
                      const selectedFixture = fixture.fixture_key === selected?.fixture_key;
                      const maximum = Math.max(fixture.home_win, fixture.draw, fixture.away_win);
                      return (
                        <button type="button" key={fixture.fixture_key} onClick={() => setSelectedKey(fixture.fixture_key)} aria-pressed={selectedFixture} className={`group w-full rounded-2xl border p-4 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#b8ff3d] ${selectedFixture ? "border-[#b8ff3d]/40 bg-[#b8ff3d]/[0.07]" : "border-white/8 bg-white/[0.025] hover:border-white/15 hover:bg-white/[0.045]"}`}>
                          <div className="flex items-center justify-between gap-3 text-xs text-slate-500"><span>{fixtureDate(fixture.kickoff_utc)}</span><span>GW {fixture.matchday ?? "—"}</span></div>
                          <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-center gap-3"><p className="truncate font-bold text-white">{fixture.home_team}</p><span className="text-xs font-semibold text-slate-600">VS</span><p className="truncate text-right font-bold text-white">{fixture.away_team}</p></div>
                          <div className="mt-4 flex items-center justify-between gap-4"><span className="text-sm text-slate-400">{outcomeLabel(fixture)} favored</span><span className="font-mono text-sm font-bold text-[#b8ff3d]">{percentage(maximum)}</span></div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </section>

              <section className="px-5 py-7 sm:px-8 lg:px-10">
                {selected ? (
                  <div className="mx-auto max-w-3xl">
                    <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500">Model selection</p><p className="mt-2 text-sm text-slate-400">{fixtureDate(selected.kickoff_utc)}</p></div><span className="rounded-full border border-[#b8ff3d]/20 bg-[#b8ff3d]/10 px-3 py-1.5 text-xs font-bold text-[#b8ff3d]">PRE-MATCH SNAPSHOT</span></div>
                    <div className="mt-8 grid grid-cols-[1fr_auto_1fr] items-center gap-4 text-center">
                      <div><p className="text-xl font-black tracking-tight text-white sm:text-3xl">{selected.home_team}</p><p className="mt-2 text-sm text-slate-500">Home</p></div>
                      <div className="rounded-2xl border border-white/10 bg-white/[0.035] px-5 py-4"><p className="font-mono text-3xl font-black text-white">{selected.predicted_score}</p><p className="mt-1 text-xs uppercase tracking-widest text-slate-600">likely score</p></div>
                      <div><p className="text-xl font-black tracking-tight text-white sm:text-3xl">{selected.away_team}</p><p className="mt-2 text-sm text-slate-500">Away</p></div>
                    </div>
                    <div className="mt-9 rounded-2xl border border-white/8 bg-[#0c131f] p-5 sm:p-6">
                      <div className="mb-6 flex items-center justify-between gap-4"><h3 className="font-bold text-white">Outcome probability</h3><span className="text-sm text-slate-500">Pick: <strong className="text-[#b8ff3d]">{outcomeLabel(selected)}</strong></span></div>
                      <div className="space-y-5">
                        <ProbabilityRow label={`${selected.home_team} win`} value={selected.home_win} tone="[&_[data-slot=progress-indicator]]:bg-[#b8ff3d]" />
                        <ProbabilityRow label="Draw" value={selected.draw} tone="[&_[data-slot=progress-indicator]]:bg-slate-400" />
                        <ProbabilityRow label={`${selected.away_team} win`} value={selected.away_win} tone="[&_[data-slot=progress-indicator]]:bg-sky-400" />
                      </div>
                    </div>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2">
                      <article className="rounded-2xl border border-white/8 bg-white/[0.025] p-5"><div className="flex items-center gap-2 text-sm text-slate-500"><Target className="size-4 text-[#b8ff3d]" /> Expected goals</div><div className="mt-5 flex items-end justify-between gap-6"><div><p className="font-mono text-3xl font-black text-white">{selected.expected_home_goals.toFixed(2)}</p><p className="mt-1 text-xs text-slate-600">{selected.home_team}</p></div><div className="text-right"><p className="font-mono text-3xl font-black text-white">{selected.expected_away_goals.toFixed(2)}</p><p className="mt-1 text-xs text-slate-600">{selected.away_team}</p></div></div></article>
                      <article className="rounded-2xl border border-white/8 bg-white/[0.025] p-5"><div className="flex items-center gap-2 text-sm text-slate-500"><Database className="size-4 text-[#b8ff3d]" /> Prediction integrity</div><p className="mt-5 font-bold text-white">Locked before kickoff</p><p className="mt-2 text-sm leading-6 text-slate-500">Later fixture syncs cannot overwrite this probability snapshot.</p></article>
                    </div>
                  </div>
                ) : null}
              </section>
            </div>
          </TabsContent>

          <TabsContent value="history" className="m-0 px-5 py-7 sm:px-8">
            <div className="mb-6"><p className="text-xs font-bold uppercase tracking-[0.2em] text-[#b8ff3d]">Audit trail</p><h2 className="mt-2 text-2xl font-black text-white">Prediction history</h2><p className="mt-2 text-sm text-slate-500">Every probability is stored before kickoff and graded once.</p></div>
            <div className="overflow-hidden rounded-2xl border border-white/8 bg-white/[0.02]">
              <Table>
                <TableHeader className="bg-white/[0.035]"><TableRow className="border-white/8 hover:bg-transparent"><TableHead className="px-4 text-slate-500">Fixture</TableHead><TableHead className="text-slate-500">Model pick</TableHead><TableHead className="text-slate-500">Probability</TableHead><TableHead className="text-slate-500">Result</TableHead><TableHead className="text-slate-500">Log loss</TableHead><TableHead className="px-4 text-right text-slate-500">Grade</TableHead></TableRow></TableHeader>
                <TableBody>
                  {history.map((prediction) => {
                    const selectedProbability = prediction.predicted_outcome === "home_win" ? prediction.home_win : prediction.predicted_outcome === "away_win" ? prediction.away_win : prediction.draw;
                    return (
                      <TableRow key={prediction.fixture_key} className="border-white/7 hover:bg-white/[0.03]">
                        <TableCell className="px-4 py-4"><p className="font-semibold text-white">{prediction.home_team} <span className="text-slate-600">vs</span> {prediction.away_team}</p><p className="mt-1 text-xs text-slate-600">{fixtureDate(prediction.kickoff_utc)}</p></TableCell>
                        <TableCell className="text-slate-300">{outcomeLabel(prediction)}</TableCell><TableCell className="font-mono font-bold text-white">{percentage(selectedProbability)}</TableCell><TableCell className="font-mono text-slate-300">{prediction.home_goals === null ? "—" : `${prediction.home_goals}–${prediction.away_goals}`}</TableCell><TableCell className="font-mono text-slate-400">{metric(prediction.log_loss)}</TableCell><TableCell className="px-4 text-right"><ResultMark correct={prediction.correct} /></TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          <TabsContent value="model" className="m-0 px-5 py-7 sm:px-8">
            <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr]">
              <section>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#b8ff3d]">Live scorecard</p><h2 className="mt-2 text-2xl font-black text-white">Model record</h2>
                <div className="mt-6 grid gap-4 sm:grid-cols-3">
                  {[{ label: "Accuracy", value: percentage(record.accuracy), detail: `${record.graded} graded fixtures` }, { label: "Log loss", value: metric(record.log_loss), detail: "lower is better" }, { label: "Brier score", value: metric(record.brier_score), detail: "probability error" }].map((item) => (
                    <article key={item.label} className="rounded-2xl border border-white/8 bg-white/[0.025] p-5"><p className="text-sm text-slate-500">{item.label}</p><p className="mt-3 font-mono text-3xl font-black text-white">{item.value}</p><p className="mt-2 text-xs text-slate-600">{item.detail}</p></article>
                  ))}
                </div>
                <div className="mt-4 rounded-2xl border border-white/8 bg-[#0c131f] p-5"><div className="flex items-center justify-between gap-4"><span className="text-sm text-slate-400">Outcome accuracy</span><span className="font-mono font-bold text-[#b8ff3d]">{percentage(record.accuracy)}</span></div><Progress value={gradedAccuracy * 100} className="mt-4 h-3 bg-white/8" /></div>
              </section>
              <section className="rounded-2xl border border-white/8 bg-white/[0.025] p-5 sm:p-6">
                <div className="flex items-center gap-2 text-sm text-slate-500"><Activity className="size-4 text-[#b8ff3d]" /> Production model</div><h3 className="mt-4 text-xl font-black text-white">V4 probability ensemble</h3>
                <div className="mt-6 space-y-5"><div><div className="flex justify-between text-sm"><span className="text-slate-400">Poisson goal model</span><span className="font-mono font-bold text-white">75%</span></div><Progress value={75} className="mt-2 bg-white/8" /></div><div><div className="flex justify-between text-sm"><span className="text-slate-400">Calibrated XGBoost</span><span className="font-mono font-bold text-white">25%</span></div><Progress value={25} className="mt-2 bg-white/8 [&_[data-slot=progress-indicator]]:bg-sky-400" /></div></div>
                <div className="mt-7 border-t border-white/8 pt-5"><p className="text-sm font-semibold text-white">Held-out 2025/26 evaluation</p><dl className="mt-4 grid grid-cols-2 gap-4 text-sm"><div><dt className="text-slate-600">Accuracy</dt><dd className="mt-1 font-mono text-slate-300">48.9%</dd></div><div><dt className="text-slate-600">Log loss</dt><dd className="mt-1 font-mono text-slate-300">1.0277</dd></div><div><dt className="text-slate-600">Walk-forward</dt><dd className="mt-1 font-mono text-slate-300">0.9605</dd></div><div><dt className="text-slate-600">Training span</dt><dd className="mt-1 font-mono text-slate-300">2000–2025</dd></div></dl></div>
              </section>
            </div>
          </TabsContent>
        </Tabs>

        <footer className="flex flex-col gap-2 border-t border-white/8 px-5 py-5 text-xs text-slate-600 sm:flex-row sm:items-center sm:justify-between sm:px-8"><span>Football data provided by the Football-Data.org API · Leak-free rolling features · No bookmaker odds</span><span className="font-mono">V4 ensemble / V8 production</span></footer>
      </div>
    </main>
  );
}
