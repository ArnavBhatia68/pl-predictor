import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const baseUrl = process.env.BACKEND_API_URL?.replace(/\/$/, "");
  if (!baseUrl) {
    return NextResponse.json(
      { error: "BACKEND_API_URL is not configured" },
      { status: 503 },
    );
  }

  try {
    const [fixturesResponse, predictionsResponse, recordResponse] = await Promise.all([
      fetch(`${baseUrl}/fixtures/upcoming?limit=20`, { cache: "no-store" }),
      fetch(`${baseUrl}/live-predictions?limit=100`, { cache: "no-store" }),
      fetch(`${baseUrl}/season-record`, { cache: "no-store" }),
    ]);

    if (!fixturesResponse.ok || !predictionsResponse.ok || !recordResponse.ok) {
      return NextResponse.json({ error: "Backend API unavailable" }, { status: 502 });
    }

    const [fixtures, predictions, record] = await Promise.all([
      fixturesResponse.json() as Promise<{ fixtures: unknown[] }>,
      predictionsResponse.json() as Promise<{ predictions: unknown[] }>,
      recordResponse.json() as Promise<Record<string, unknown>>,
    ]);

    return NextResponse.json(
      {
        fixtures: fixtures.fixtures,
        predictions: predictions.predictions,
        record,
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return NextResponse.json({ error: "Backend API unavailable" }, { status: 502 });
  }
}
