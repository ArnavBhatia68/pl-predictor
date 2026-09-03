from __future__ import annotations

import argparse

from .data import build_match_dataset
from .features import build_feature_dataset
from .stat_models import train_stat_models
from .train import train_models
from .v2 import run_v2
from .v3 import run_v3
from .v4 import run_v4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Premier League prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and normalize season CSVs")
    download.add_argument("--start-season", type=int, default=2000)
    download.add_argument("--end-season", type=int, default=2026)
    download.add_argument("--force", action="store_true")
    download.add_argument(
        "--refresh-current",
        action="store_true",
        help="Redownload only the final requested season while reusing historical files",
    )

    subparsers.add_parser("features", help="Build leak-free pre-match features")

    train = subparsers.add_parser("train", help="Select, train, and evaluate models")
    train.add_argument("--validation-season", type=int, default=2023)
    train.add_argument("--test-seasons", type=int, nargs="+", default=[2024, 2025])
    train.add_argument("--exclude-seasons", type=int, nargs="*", default=[2026])

    v2 = subparsers.add_parser("v2", help="Run walk-forward V2 tuning and calibration")
    v2.add_argument("--first-fold", type=int, default=2018)
    v2.add_argument("--last-fold", type=int, default=2023)
    v2.add_argument("--calibration-season", type=int, default=2024)
    v2.add_argument("--test-season", type=int, default=2025)

    v3 = subparsers.add_parser("v3", help="Run Poisson expected-goals outcome modeling")
    v3.add_argument("--first-fold", type=int, default=2018)
    v3.add_argument("--last-fold", type=int, default=2023)
    v3.add_argument("--calibration-season", type=int, default=2024)
    v3.add_argument("--test-season", type=int, default=2025)

    v4 = subparsers.add_parser("v4", help="Tune and evaluate the V2/V3 probability ensemble")
    v4.add_argument("--first-fold", type=int, default=2018)
    v4.add_argument("--last-fold", type=int, default=2023)
    v4.add_argument("--test-season", type=int, default=2025)
    v4.add_argument("--production-season", type=int, default=2026)

    stats = subparsers.add_parser("stats", help="Train V11 detailed match-stat models")
    stats.add_argument("--first-fold", type=int, default=2022)
    stats.add_argument("--last-fold", type=int, default=2024)
    stats.add_argument("--test-season", type=int, default=2025)
    stats.add_argument("--production-season", type=int, default=2026)

    sync = subparsers.add_parser(
        "sync-fixtures",
        help="Fetch fixtures, save pre-match predictions, and grade completed matches",
    )
    sync.add_argument("--days-back", type=int, default=3)
    sync.add_argument("--days-ahead", type=int, default=14)

    refresh = subparsers.add_parser(
        "refresh-live",
        help="Refresh current match data, sync fixtures, and grade predictions",
    )
    refresh.add_argument("--days-back", type=int, default=3)
    refresh.add_argument("--days-ahead", type=int, default=14)

    record = subparsers.add_parser("record", help="Show live prediction performance")
    record.add_argument("--limit", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        build_match_dataset(
            args.start_season,
            args.end_season,
            force=args.force,
            refresh_current=args.refresh_current,
        )
    elif args.command == "features":
        build_feature_dataset()
    elif args.command == "train":
        train_models(
            validation_season=args.validation_season,
            test_seasons=tuple(args.test_seasons),
            exclude_seasons=tuple(args.exclude_seasons),
        )
    elif args.command == "v2":
        run_v2(
            validation_seasons=tuple(range(args.first_fold, args.last_fold + 1)),
            calibration_season=args.calibration_season,
            test_season=args.test_season,
        )
    elif args.command == "v3":
        run_v3(
            validation_seasons=tuple(range(args.first_fold, args.last_fold + 1)),
            calibration_season=args.calibration_season,
            test_season=args.test_season,
        )
    elif args.command == "v4":
        run_v4(
            validation_seasons=tuple(range(args.first_fold, args.last_fold + 1)),
            test_season=args.test_season,
            production_season=args.production_season,
        )
    elif args.command == "stats":
        train_stat_models(
            validation_seasons=tuple(range(args.first_fold, args.last_fold + 1)),
            test_season=args.test_season,
            production_season=args.production_season,
        )
    elif args.command == "sync-fixtures":
        from .fixtures import FootballDataOrgProvider
        from .service import PredictionService
        from .tracking import FixtureTracker, PredictionStore

        tracker = FixtureTracker(PredictionService.from_paths(), PredictionStore())
        print(
            tracker.sync(
                FootballDataOrgProvider(),
                days_back=args.days_back,
                days_ahead=args.days_ahead,
            )
        )
    elif args.command == "refresh-live":
        from .api import refresh_live_data

        print(refresh_live_data(args.days_back, args.days_ahead))
    elif args.command == "record":
        from .tracking import PredictionStore

        store = PredictionStore()
        print(store.record())
        for prediction in store.prediction_history(args.limit):
            print(prediction)


if __name__ == "__main__":
    main()
