"""Action Plan API for Sprint 5 case workspace workflows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.action_plan import ActionPlan, ActionPlanStatus, ActionType
from backend.models.audit_case import AuditCase

router = APIRouter(tags=["action-plans"])


class ActionPlanCreateRequest(BaseModel):
    owner_email: EmailStr
    owner_department: Optional[str] = Field(default=None, max_length=100)
    action_type: ActionType
    deadline: datetime
    notes: Optional[str] = Field(default=None, max_length=10000)
    estimated_recovery_usd: Optional[float] = Field(default=None, ge=0)


class ActionPlanUpdateStatusRequest(BaseModel):
    status: ActionPlanStatus
    dollars_saved: Optional[float] = Field(default=None, ge=0)
    resolution_notes: Optional[str] = Field(default=None, max_length=10000)


class ActionPlanResponse(BaseModel):
    id: uuid.UUID
    case_id: uuid.UUID
    owner_email: str
    owner_department: Optional[str]
    action_type: ActionType
    deadline: str
    notes: Optional[str]
    dollars_saved: Optional[float]
    estimated_recovery_usd: Optional[float]
    status: ActionPlanStatus
    completed_at: Optional[str]
    resolution_notes: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class ActionPlanListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ActionPlanResponse]


def _to_float(value: Decimal | None) -> Optional[float]:
    return float(value) if value is not None else None


def _to_response(plan: ActionPlan) -> ActionPlanResponse:
    return ActionPlanResponse(
        id=plan.id,
        case_id=plan.case_id,
        owner_email=plan.owner_email,
        owner_department=plan.owner_department,
        action_type=plan.action_type,
        deadline=plan.deadline.isoformat(),
        notes=plan.notes,
        dollars_saved=_to_float(plan.dollars_saved),
        estimated_recovery_usd=_to_float(plan.estimated_recovery_usd),
        status=plan.status,
        completed_at=plan.completed_at.isoformat() if plan.completed_at else None,
        resolution_notes=plan.resolution_notes,
        created_at=plan.created_at.isoformat() if plan.created_at else None,
        updated_at=plan.updated_at.isoformat() if plan.updated_at else None,
    )


@router.post(
    "/cases/{case_id}/action-plan",
    response_model=ActionPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create action plan for an audit case",
)
async def create_action_plan(
    case_id: uuid.UUID,
    body: ActionPlanCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ActionPlanResponse:
    case = await db.scalar(select(AuditCase).where(AuditCase.id == case_id))
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit case {case_id} not found",
        )

    plan = ActionPlan(
        case_id=case_id,
        owner_email=str(body.owner_email),
        owner_department=body.owner_department,
        action_type=body.action_type,
        deadline=body.deadline,
        notes=body.notes,
        estimated_recovery_usd=body.estimated_recovery_usd,
        status=ActionPlanStatus.PENDING,
    )
    db.add(plan)
    await db.flush()
    await db.commit()

    return _to_response(plan)


@router.patch(
    "/action-plans/{plan_id}/status",
    response_model=ActionPlanResponse,
    summary="Update action plan execution status",
)
async def update_action_plan_status(
    plan_id: uuid.UUID,
    body: ActionPlanUpdateStatusRequest,
    db: AsyncSession = Depends(get_db),
) -> ActionPlanResponse:
    plan = await db.scalar(select(ActionPlan).where(ActionPlan.id == plan_id))
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Action plan {plan_id} not found",
        )

    plan.status = body.status
    plan.resolution_notes = body.resolution_notes

    if body.status == ActionPlanStatus.COMPLETED:
        plan.completed_at = datetime.now(timezone.utc)
        if body.dollars_saved is not None:
            plan.dollars_saved = body.dollars_saved
    else:
        if body.dollars_saved is not None:
            plan.dollars_saved = body.dollars_saved

    await db.flush()
    await db.commit()

    return _to_response(plan)


@router.get(
    "/action-plans",
    response_model=ActionPlanListResponse,
    summary="List action plans",
)
async def list_action_plans(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[ActionPlanStatus] = Query(None, alias="status"),
    owner_email: Optional[str] = Query(None),
    case_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> ActionPlanListResponse:
    stmt = select(ActionPlan).order_by(ActionPlan.created_at.desc())
    if status_filter:
        stmt = stmt.where(ActionPlan.status == status_filter)
    if owner_email:
        stmt = stmt.where(ActionPlan.owner_email == owner_email)
    if case_id:
        stmt = stmt.where(ActionPlan.case_id == case_id)

    from sqlalchemy import func
    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))

    rows = await db.execute(stmt.offset((page - 1) * page_size).limit(page_size))
    items = rows.scalars().all()

    return ActionPlanListResponse(
        total=int(total or 0),
        page=page,
        page_size=page_size,
        items=[_to_response(plan) for plan in items],
    )
