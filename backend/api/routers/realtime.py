"""Real-time stream endpoints for live frontend dashboards."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select

from backend.database import AsyncSessionLocal
from backend.models.action_plan import ActionPlan, ActionPlanStatus
from backend.models.audit_case import AuditCase, AuditCaseStatus
from backend.models.transaction import Transaction

router = APIRouter(prefix="/realtime", tags=["realtime"])


async def _collect_snapshot() -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        total_transactions = int(await db.scalar(select(func.count(Transaction.id))) or 0)
        audited_transactions = int(await db.scalar(select(func.count(AuditCase.id))) or 0)

        status_rows = await db.execute(
            select(AuditCase.status, func.count(AuditCase.id)).group_by(AuditCase.status)
        )
        status_counts = defaultdict(int)
        for status, count in status_rows.all():
            status_counts[status] = int(count)

        completed_count = int(
            await db.scalar(
                select(func.count(ActionPlan.id)).where(ActionPlan.status == ActionPlanStatus.COMPLETED)
            )
            or 0
        )

        total_saved = float(
            await db.scalar(
                select(func.coalesce(func.sum(ActionPlan.dollars_saved), 0)).where(
                    ActionPlan.status == ActionPlanStatus.COMPLETED
                )
            )
            or 0
        )

        avg_saved = float(
            await db.scalar(
                select(func.coalesce(func.avg(ActionPlan.dollars_saved), 0)).where(
                    ActionPlan.status == ActionPlanStatus.COMPLETED
                )
            )
            or 0
        )

        latest_cases_rows = await db.execute(
            select(AuditCase)
            .order_by(AuditCase.updated_at.desc())
            .limit(8)
        )
        latest_case_entities = latest_cases_rows.scalars().all()
        latest_cases: List[Dict[str, Any]] = []
        for case in latest_case_entities:
            latest_cases.append(
                {
                    "id": str(case.id),
                    "risk_level": case.risk_level,
                    "status": case.status.value,
                    "ml_score": case.ml_score,
                    "groq_escalated": case.groq_escalated,
                    "updated_at": case.updated_at.isoformat() if case.updated_at else None,
                }
            )

        latest_plans_rows = await db.execute(
            select(ActionPlan)
            .order_by(ActionPlan.updated_at.desc())
            .limit(8)
        )

        recent_events: List[Dict[str, Any]] = []
        for case in latest_case_entities:
            event_at = case.updated_at or case.created_at
            if not event_at:
                continue
            recent_events.append(
                {
                    "id": f"case:{case.id}:{event_at.isoformat()}",
                    "timestamp": event_at.isoformat(),
                    "entity": "CASE",
                    "title": f"Case {str(case.id)[:8]} updated",
                    "detail": f"Risk={case.risk_level or 'UNKNOWN'} · Status={case.status.value}",
                    "severity": (case.risk_level or "LOW").upper(),
                }
            )

        for plan in latest_plans_rows.scalars().all():
            event_at = plan.updated_at or plan.created_at
            if not event_at:
                continue
            recent_events.append(
                {
                    "id": f"plan:{plan.id}:{event_at.isoformat()}",
                    "timestamp": event_at.isoformat(),
                    "entity": "ACTION_PLAN",
                    "title": f"Action plan {str(plan.id)[:8]} {plan.status.value.lower()}",
                    "detail": f"Type={plan.action_type.value} · Owner={plan.owner_email}",
                    "severity": "MEDIUM" if plan.status == ActionPlanStatus.COMPLETED else "LOW",
                }
            )

        recent_events.sort(key=lambda event: event["timestamp"], reverse=True)
        recent_events = recent_events[:12]

        coverage_pct = round((audited_transactions / total_transactions) * 100, 2) if total_transactions else 0.0

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coverage": {
                "total_transactions": total_transactions,
                "audited_transactions": audited_transactions,
                "open_cases": status_counts[AuditCaseStatus.OPEN],
                "in_review_cases": status_counts[AuditCaseStatus.IN_REVIEW],
                "closed_cases": status_counts[AuditCaseStatus.CLOSED],
                "audit_coverage_pct": coverage_pct,
            },
            "roi": {
                "completed_action_plans": completed_count,
                "total_dollars_saved": total_saved,
                "average_dollars_saved": avg_saved,
            },
            "recent_cases": latest_cases,
            "recent_events": recent_events,
        }


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    interval_seconds: int = Query(default=4, ge=1, le=30),
) -> None:
    """Pushes live KPI and case snapshots to connected frontend clients."""
    await websocket.accept()

    try:
        while True:
            snapshot = await _collect_snapshot()
            await websocket.send_json({"type": "snapshot", "payload": snapshot})
            await asyncio.sleep(interval_seconds)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
