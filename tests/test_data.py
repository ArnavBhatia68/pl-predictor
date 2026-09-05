import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pl_predictor.data import _current_season_start, download_season, season_code, season_label


class DataTests(unittest.TestCase):
    def test_season_formatting(self) -> None:
        self.assertEqual(season_code(2000), "0001")
        self.assertEqual(season_code(2026), "2627")
        self.assertEqual(season_label(2000), "2000/01")

    def test_current_season_boundary(self) -> None:
        self.assertEqual(_current_season_start(date(2026, 9, 1)), 2026)
        self.assertEqual(_current_season_start(date(2027, 2, 1)), 2026)

    def test_forced_refresh_uses_cached_season_when_provider_is_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            cached = raw_dir / "2627.csv"
            cached.write_text("cached-data", encoding="utf-8")

            with patch(
                "pl_predictor.data._fetch_bytes",
                side_effect=RuntimeError("HTTP Error 503: Service Temporarily Unavailable"),
            ):
                result = download_season(2026, raw_dir=raw_dir, force=True)

            self.assertEqual(result, cached)
            self.assertEqual(cached.read_text(encoding="utf-8"), "cached-data")

    def test_refresh_still_fails_when_no_cached_season_exists(self) -> None:
        with TemporaryDirectory() as directory, patch(
            "pl_predictor.data._fetch_bytes",
            side_effect=RuntimeError("provider unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            download_season(2026, raw_dir=Path(directory), force=True)


if __name__ == "__main__":
    unittest.main()
