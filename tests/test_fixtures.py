import unittest
from datetime import date
from unittest.mock import patch

import httpx

from pl_predictor.fixtures import FootballDataOrgProvider


class FixtureProviderTests(unittest.TestCase):
    def test_football_data_response_is_normalized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Auth-Token"], "secret")
            self.assertEqual(request.url.params["dateFrom"], "2026-09-01")
            return httpx.Response(
                200,
                json={
                    "matches": [
                        {
                            "id": 123,
                            "utcDate": "2026-09-12T14:00:00Z",
                            "status": "TIMED",
                            "matchday": 4,
                            "season": {"startDate": "2026-08-08"},
                            "homeTeam": {"name": "Chelsea FC"},
                            "awayTeam": {"name": "Arsenal FC"},
                            "score": {"fullTime": {"home": None, "away": None}},
                        }
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = FootballDataOrgProvider(token="secret", client=client)
        fixtures = provider.fetch_matches(
            date(2026, 9, 1),
            date(2026, 9, 15),
        )
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0].key, "football-data.org:123")
        self.assertEqual(fixtures[0].season_start, 2026)
        self.assertEqual(fixtures[0].home_team, "Chelsea FC")
        self.assertEqual(fixtures[0].kickoff_utc.isoformat(), "2026-09-12T14:00:00+00:00")
        client.close()

    def test_token_is_required(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            FootballDataOrgProvider(token="")

    def test_scorers_are_normalized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v4/competitions/PL/scorers")
            return httpx.Response(
                200,
                json={
                    "scorers": [
                        {
                            "player": {"id": 7, "name": "Example Forward", "position": "Offence"},
                            "team": {"name": "Chelsea FC"},
                            "goals": 4,
                            "assists": 2,
                            "penalties": 1,
                        }
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = FootballDataOrgProvider(token="secret", client=client)
        scorer = provider.fetch_scorers()[0]
        self.assertEqual(scorer["name"], "Example Forward")
        self.assertEqual(scorer["team"], "Chelsea FC")
        self.assertEqual(scorer["goals"], 4)
        client.close()

    def test_provider_error_is_safe_and_actionable(self) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(429, request=request))
        )
        provider = FootballDataOrgProvider(token="secret", client=client)
        with self.assertRaisesRegex(RuntimeError, "Fixture provider request failed"):
            provider.fetch_matches(date(2026, 9, 1), date(2026, 9, 15))
        client.close()


if __name__ == "__main__":
    unittest.main()
