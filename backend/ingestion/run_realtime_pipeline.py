"""CLI entrypoint for near-realtime multi-source ingestion pipeline."""

from __future__ import annotations

import argparse
import asyncio

from backend.ingestion.realtime_pipeline import NearRealtimeIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run near-realtime procurement ingestion pipeline")
    parser.add_argument("--interval", type=int, default=300, help="poll interval in seconds (default: 300)")
    parser.add_argument("--limit", type=int, default=100, help="max records per source per cycle")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pipeline = NearRealtimeIngestionPipeline(poll_interval_seconds=args.interval)
    if args.once:
        await pipeline.run_once(limit_per_source=args.limit)
        return
    await pipeline.run_forever(limit_per_source=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
