export type CaseStatus = 'OPEN' | 'IN_REVIEW' | 'CLOSED'
export type ActionType =
  | 'CLAWBACK'
  | 'PAYMENT_HALT'
  | 'VENDOR_REVIEW'
  | 'CONTRACT_RENEGOTIATION'
  | 'TEMPLATE_UPDATE'
  | 'ESCALATE'
  | 'DISMISS'

export type CaseSummary = {
  id: string
  transaction_id: string
  ml_score: number
  status: CaseStatus
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW' | null
  groq_escalated: boolean | null
  llm_verdict: 'FRAUD' | 'SUSPICIOUS' | 'NORMAL' | 'INCONCLUSIVE' | null
  confidence: number | null
  estimated_impact_usd: number | null
  created_at: string | null
  updated_at?: string | null
}

export type CaseDetail = CaseSummary & {
  shap_summary: string | null
  groq_verdict: { escalate: boolean; reason: string; risk_level: string } | null
  gemini_report: {
    verdict: string
    confidence: number
    rationale: string
    violated_clause?: string
    cited_clause_text?: string
  } | null
  contract_clause_cited: string | null
  violated_clause_id: string | null
  auditor_notes: string | null
}

export type ContractItem = {
  id: string
  title: string
  vendor_id: string
  status: string
  is_indexed: boolean
  chunk_count: number
  created_at?: string
}

export type Coverage = {
  total_transactions: number
  audited_transactions: number
  open_cases: number
  in_review_cases: number
  closed_cases: number
  audit_coverage_pct: number
}

export type Roi = {
  completed_action_plans: number
  total_dollars_saved: number
  average_dollars_saved: number
}

export type RealtimeCase = {
  id: string
  risk_level: 'HIGH' | 'MEDIUM' | 'LOW' | null
  status: CaseStatus
  ml_score: number
  groq_escalated: boolean | null
  updated_at: string | null
}

export type RealtimeSnapshot = {
  timestamp: string
  coverage: Coverage
  roi: Roi
  recent_cases: RealtimeCase[]
  recent_events: RealtimeEvent[]
}

export type PipelineMetrics = {
  total_transactions: number
  scored_transactions: number
  unscored_transactions: number
  score_coverage_pct: number
  audit_cases: number
  audit_coverage_pct: number
}

export type SourceMetricsRow = {
  source_system: string
  agency_family: string
  category_family: string
  total_transactions: number
  scored_transactions: number
  audit_cases: number
  escalated_cases: number
  high_risk_cases: number
  avg_ml_score: number
  score_coverage_pct: number
}

export type OrchestrationRunSummary = {
  scanned: number
  scored: number
  cases_created: number
  cases_updated: number
  llm_triaged: number
  llm_deep_audited: number
  high_risk: number
  medium_risk: number
  low_risk: number
  started_at: string
  completed_at: string | null
}

export type RealtimeEvent = {
  id: string
  timestamp: string
  entity: 'CASE' | 'ACTION_PLAN'
  title: string
  detail: string
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
}

export type ActionPlan = {
  id: string
  case_id: string
  owner_email: string
  owner_department: string | null
  action_type: ActionType
  deadline: string
  notes: string | null
  dollars_saved: number | null
  estimated_recovery_usd: number | null
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'CANCELLED'
  completed_at: string | null
  resolution_notes: string | null
  created_at: string | null
  updated_at: string | null
}

type RequestOptions = {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'
  body?: unknown
  signal?: AbortSignal
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function toUrl(path: string) {
  if (!API_BASE) return path
  return `${API_BASE}${path}`
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 20_000)

  try {
    const response = await fetch(toUrl(path), {
      method: options.method ?? 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal ?? controller.signal,
    })

    if (!response.ok) {
      const details = await response.text()
      throw new Error(details || `HTTP ${response.status}`)
    }

    return (await response.json()) as T
  } finally {
    clearTimeout(timeout)
  }
}

export async function fetchCoverage() {
  return request<Coverage>('/api/v1/metrics/coverage')
}

export async function fetchRoi() {
  return request<Roi>('/api/v1/metrics/roi')
}

export async function fetchPipelineMetrics() {
  return request<PipelineMetrics>('/api/v1/metrics/pipeline')
}

export async function fetchSourceMetrics(limit = 20) {
  return request<SourceMetricsRow[]>(`/api/v1/metrics/sources?limit=${limit}`)
}

export async function fetchCases(params: Record<string, string | number>) {
  const query = new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)]))
  return request<{ items: CaseSummary[]; total: number }>(`/api/v1/cases?${query.toString()}`)
}

export async function fetchCaseDetail(caseId: string) {
  return request<CaseDetail>(`/api/v1/cases/${caseId}`)
}

export async function fetchContracts(params: Record<string, string | number>) {
  const query = new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)]))
  return request<{ items: ContractItem[]; total: number }>(`/api/v1/contracts?${query.toString()}`)
}

export async function searchContracts(params: { q: string }) {
  const query = new URLSearchParams(params)
  return request<{ query: string; results: Array<{ text: string; score: number }> }>(
    `/api/v1/contracts/search?${query.toString()}`,
  )
}

export async function createActionPlan(input: {
  caseId: string
  payload: {
    owner_email: string
    owner_department?: string
    action_type: ActionType
    deadline: string
    estimated_recovery_usd?: number
    notes?: string
  }
}) {
  return request<ActionPlan>(`/api/v1/cases/${input.caseId}/action-plan`, {
    method: 'POST',
    body: input.payload,
  })
}

export async function fetchActionPlans(params: Record<string, string | number>) {
  const query = new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)]))
  return request<{ items: ActionPlan[]; total: number }>(`/api/v1/action-plans?${query.toString()}`)
}

export async function runOrchestrationNow(input: { batch_size: number; run_llm?: boolean }) {
  const query = new URLSearchParams({
    batch_size: String(input.batch_size),
    run_llm: String(Boolean(input.run_llm)),
  })
  return request<OrchestrationRunSummary>(`/api/v1/orchestration/run?${query.toString()}`, {
    method: 'POST',
  })
}
