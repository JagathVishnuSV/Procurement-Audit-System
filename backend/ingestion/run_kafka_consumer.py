"""CLI entrypoint for raw transaction Kafka consumer."""

from __future__ import annotations

import asyncio

from backend.ingestion.kafka_consumer import ProcurementKafkaConsumer


async def main() -> None:
    consumer = ProcurementKafkaConsumer()
    await consumer.start()
    try:
        await consumer.run_forever()
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
