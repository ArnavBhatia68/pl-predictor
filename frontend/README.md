# PL Predictor Dashboard

Next.js/Vinext frontend for the Premier League prediction API.

## Local setup

```bash
npm run install:ci
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` to the FastAPI origin. Without it, the dashboard intentionally
loads representative demo data so the deployed interface remains reviewable before the Python
backend is hosted.

The interface consumes `GET /fixtures/upcoming`, `GET /live-predictions`, and
`GET /season-record`. It includes the upcoming gameweek, stored probability snapshots, expected
goals, prediction history, live grading metrics, and V4 ensemble details.
