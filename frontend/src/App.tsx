import clsx from 'clsx'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'
import {
  createActionPlan,
  fetchActionPlans,
  fetchCaseDetail,
  fetchCases,
  fetchContracts,
  fetchCoverage,
  fetchPipelineMetrics,
  fetchRoi,
  fetchSourceMetrics,
  runOrchestrationNow,
  searchContracts,
  type ActionType,
  type ActionPlan,
  type CaseDetail,
  type CaseSummary,
  type ContractItem,
  type Coverage,
  type PipelineMetrics,
  type RealtimeEvent,
  type Roi,
  type RealtimeSnapshot,
  type SourceMetricsRow,
} from './api/client'
import './App.css'

const money = (value?: number | null) =>
  typeof value === 'number' ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '—'

const percent = (value?: number | null) => {
  if (typeof value !== 'number') return '—'
  if (value === 0) return '0%'
  if (value < 0.01) return '<0.01%'
  return `${value.toFixed(2)}%`
}

const isNotifiableSeverity = (severity: RealtimeEvent['severity']) =>
  severity === 'HIGH' || severity === 'MEDIUM'

function App() {
  const queryClient = useQueryClient()
  const [liveSnapshot, setLiveSnapshot] = useState<RealtimeSnapshot | null>(null)
  const [streamState, setStreamState] = useState<'CONNECTING' | 'LIVE' | 'OFFLINE'>('CONNECTING')
  const [toasts, setToasts] = useState<Array<{ id: string; title: string; detail: string; severity: string }>>([])
  const seenEventIds = useRef<Set<string>>(new Set())
  const primed = useRef(false)
  const reconnectTimer = useRef<number | null>(null)

  useEffect(() => {
    let socket: WebSocket | null = null
    let active = true

    const connect = () => {
      if (!active) return
      setStreamState('CONNECTING')

      const configuredBase = import.meta.env.VITE_API_BASE_URL
      const base = configuredBase
        ? new URL(configuredBase)
        : new URL('http://127.0.0.1:8000')
      const protocol = base.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${base.host}/api/v1/realtime/stream`

      socket = new WebSocket(wsUrl)

      socket.onopen = () => {
        if (!active) return
        setStreamState('LIVE')
      }

      socket.onmessage = (event) => {
        if (!active) return
        try {
          const message = JSON.parse(event.data as string)
          if (message?.type !== 'snapshot' || !message?.payload) return
          const payload = message.payload as RealtimeSnapshot
          setLiveSnapshot(payload)
          queryClient.setQueryData(['coverage'], payload.coverage)
          queryClient.setQueryData(['roi'], payload.roi)
          queryClient.setQueryData(['cases', 'live-recent'], { items: payload.recent_cases })

          const events = payload.recent_events ?? []
          if (!primed.current) {
            events.forEach((event) => seenEventIds.current.add(event.id))
            primed.current = true
          } else {
            const fresh = events
              .filter((event) => !seenEventIds.current.has(event.id) && isNotifiableSeverity(event.severity))
              .slice(0, 3)
            fresh.forEach((event) => seenEventIds.current.add(event.id))
            if (fresh.length) {
              setToasts((prev) => {
                const next = [
                  ...fresh.map((event) => ({
                    id: event.id,
                    title: event.title,
                    detail: event.detail,
                    severity: event.severity,
                  })),
                  ...prev,
                ]
                return next.slice(0, 6)
              })
            }
          }

          queryClient.invalidateQueries({ queryKey: ['cases'] })
        } catch {
          return
        }
      }

      socket.onerror = () => {
        if (!active) return
        setStreamState('OFFLINE')
      }

      socket.onclose = () => {
        if (!active) return
        setStreamState('OFFLINE')
        reconnectTimer.current = window.setTimeout(connect, 2500)
      }
    }

    connect()

    return () => {
      active = false
      if (reconnectTimer.current) {
        window.clearTimeout(reconnectTimer.current)
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close()
      }
    }
  }, [queryClient])

  return (
    <div className="app-root">
      <AppShell streamState={streamState} liveTimestamp={liveSnapshot?.timestamp ?? null}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage liveSnapshot={liveSnapshot} />} />
          <Route path="/cases" element={<CasesPage liveSnapshot={liveSnapshot} />} />
          <Route path="/cases/:caseId" element={<CaseWorkspacePage />} />
          <Route path="/contracts" element={<ContractsPage />} />
        </Routes>
      </AppShell>
      <div className="toast-stack" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={clsx('toast', toast.severity.toLowerCase())}>
            <strong>{toast.title}</strong>
            <p>{toast.detail}</p>
            <button type="button" onClick={() => setToasts((prev) => prev.filter((item) => item.id !== toast.id))}>
              Dismiss
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

function AppShell({
  children,
  streamState,
  liveTimestamp,
}: {
  children: ReactNode
  streamState: 'CONNECTING' | 'LIVE' | 'OFFLINE'
  liveTimestamp: string | null
}) {
  const location = useLocation()
  const navItems = [
    { label: 'Executive Dashboard', path: '/dashboard' },
    { label: 'Audit Inbox', path: '/cases' },
    { label: 'Smart CLM', path: '/contracts' },
  ]

  return (
    <div className="shell">
      <aside className="sidebar">
        <h1>Procurement Audit</h1>
        <p className="subtle">Real-time anomaly detection, triage, and forensic actioning.</p>
        <nav>
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={clsx('nav-link', location.pathname.startsWith(item.path) && 'active')}
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="sidebar-hint">
          <strong>How to use</strong>
          <p>Start in Audit Inbox → inspect case evidence → assign Action Plan → monitor ROI in dashboard.</p>
        </div>
        <div className="stream-status">
          <span className={clsx('status-dot', streamState.toLowerCase())} />
          <div>
            <strong>Live Stream: {streamState}</strong>
            <p>{liveTimestamp ? `Last update: ${new Date(liveTimestamp).toLocaleTimeString()}` : 'Waiting for first snapshot…'}</p>
          </div>
        </div>
      </aside>
      <main className="content">{children}</main>
    </div>
  )
}

function DashboardPage({ liveSnapshot }: { liveSnapshot: RealtimeSnapshot | null }) {
  const queryClient = useQueryClient()
  const [manualBatchSize, setManualBatchSize] = useState(200)

  const { data: coverage, isLoading: coverageLoading } = useQuery<Coverage>({
    queryKey: ['coverage'],
    queryFn: fetchCoverage,
    refetchInterval: 30_000,
  })
  const { data: roi, isLoading: roiLoading } = useQuery<Roi>({
    queryKey: ['roi'],
    queryFn: fetchRoi,
    refetchInterval: 30_000,
  })
  const { data: casesData } = useQuery<{ items: CaseSummary[]; total: number }>({
    queryKey: ['cases', 'dashboard'],
    queryFn: () => fetchCases({ page: 1, page_size: 40 }),
  })
  const { data: pipeline, isLoading: pipelineLoading } = useQuery<PipelineMetrics>({
    queryKey: ['pipeline'],
    queryFn: fetchPipelineMetrics,
    refetchInterval: 30_000,
  })
  const { data: sourceMetrics, isLoading: sourceMetricsLoading } = useQuery<SourceMetricsRow[]>({
    queryKey: ['source-metrics'],
    queryFn: () => fetchSourceMetrics(12),
    refetchInterval: 45_000,
  })

  const runNowMutation = useMutation({
    mutationFn: (input: { batch_size: number; run_llm?: boolean }) => runOrchestrationNow(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['coverage'] })
      queryClient.invalidateQueries({ queryKey: ['roi'] })
      queryClient.invalidateQueries({ queryKey: ['pipeline'] })
      queryClient.invalidateQueries({ queryKey: ['source-metrics'] })
      queryClient.invalidateQueries({ queryKey: ['cases'] })
    },
  })

  const statusChart = useMemo(
    () => [
      { name: 'Open', value: coverage?.open_cases ?? 0, color: '#ff6b8a' },
      { name: 'In Review', value: coverage?.in_review_cases ?? 0, color: '#ffd166' },
      { name: 'Closed', value: coverage?.closed_cases ?? 0, color: '#4fd1a1' },
    ],
    [coverage],
  )

  const topRiskCases = (casesData?.items ?? [])
    .filter((item) => item.risk_level === 'HIGH' || item.groq_escalated)
    .slice(0, 5)

  const liveRiskMix = useMemo(() => {
    const cases = liveSnapshot?.recent_cases ?? []
    if (!cases.length) return { high: 0, medium: 0, low: 0 }
    return cases.reduce(
      (acc, item) => {
        if (item.risk_level === 'HIGH') acc.high += 1
        else if (item.risk_level === 'MEDIUM') acc.medium += 1
        else acc.low += 1
        return acc
      },
      { high: 0, medium: 0, low: 0 },
    )
  }, [liveSnapshot])

  return (
    <section className="page">
      <header className="page-header">
        <h2>Executive ROI Dashboard</h2>
        <p>Live operational snapshot for risk coverage, case throughput, and financial outcome.</p>
      </header>

      <div className="kpi-grid">
        <KpiCard title="Total Transactions" value={coverageLoading ? 'Loading…' : String(coverage?.total_transactions ?? 0)} />
        <KpiCard title="Audit Coverage" value={coverageLoading ? 'Loading…' : percent(coverage?.audit_coverage_pct)} />
        <KpiCard title="Cost Savings YTD" value={roiLoading ? 'Loading…' : money(roi?.total_dollars_saved)} />
        <KpiCard title="Avg Savings / Action" value={roiLoading ? 'Loading…' : money(roi?.average_dollars_saved)} />
      </div>

      <div className="kpi-grid">
        <KpiCard
          title="Scored Transactions"
          value={pipelineLoading ? 'Loading…' : `${pipeline?.scored_transactions ?? 0}/${pipeline?.total_transactions ?? 0}`}
        />
        <KpiCard title="Scoring Coverage" value={pipelineLoading ? 'Loading…' : percent(pipeline?.score_coverage_pct)} />
        <KpiCard title="Unscored Backlog" value={pipelineLoading ? 'Loading…' : String(pipeline?.unscored_transactions ?? 0)} />
        <KpiCard title="Audit Cases" value={pipelineLoading ? 'Loading…' : String(pipeline?.audit_cases ?? 0)} />
      </div>

      <article className="panel">
        <h3>Pipeline Orchestration</h3>
        <p className="subtle">Score latest unscored transactions on demand and refresh source/risk signals.</p>
        <form
          className="toolbar"
          onSubmit={(event) => {
            event.preventDefault()
            runNowMutation.mutate({ batch_size: manualBatchSize, run_llm: false })
          }}
        >
          <input
            type="number"
            min={1}
            max={5000}
            value={manualBatchSize}
            onChange={(event) => setManualBatchSize(Number(event.target.value) || 1)}
          />
          <button type="submit" disabled={runNowMutation.isPending}>
            {runNowMutation.isPending ? 'Running…' : 'Score Latest N'}
          </button>
        </form>
        {runNowMutation.isSuccess ? (
          <p className="success">
            Batch complete: scored {runNowMutation.data.scored}, created {runNowMutation.data.cases_created}, updated {runNowMutation.data.cases_updated}.
          </p>
        ) : null}
        {runNowMutation.isError ? <p className="error">Failed to run orchestration batch. Check backend logs and retry.</p> : null}
      </article>

      <div className="panel-grid">
        <article className="panel">
          <h3>Case Distribution</h3>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statusChart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3558" />
                <XAxis dataKey="name" stroke="#b6c2e4" />
                <YAxis stroke="#b6c2e4" />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Cases" radius={[8, 8, 0, 0]}>
                  {statusChart.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <h3>Priority Queue</h3>
          {topRiskCases.length === 0 ? (
            <p className="subtle">No high-priority cases at this moment.</p>
          ) : (
            <ul className="risk-list">
              {topRiskCases.map((item: CaseSummary) => (
                <li key={item.id}>
                  <div>
                    <strong>{item.risk_level ?? 'UNKNOWN'} risk</strong>
                    <p>{item.id.slice(0, 20)}…</p>
                  </div>
                  <Link to={`/cases/${item.id}`}>Open</Link>
                </li>
              ))}
            </ul>
          )}
        </article>
      </div>

      <div className="panel-grid">
        <article className="panel">
          <h3>Live Insight Signals</h3>
          <ul className="insight-list">
            <li>
              <strong>High-Risk Ratio</strong>
              <span>
                {liveSnapshot?.recent_cases?.length
                  ? `${Math.round((liveRiskMix.high / liveSnapshot.recent_cases.length) * 100)}%`
                  : '0%'}
              </span>
            </li>
            <li>
              <strong>Escalation Momentum</strong>
              <span>
                {liveSnapshot?.recent_cases
                  ? `${liveSnapshot.recent_cases.filter((item) => item.groq_escalated).length}/${liveSnapshot.recent_cases.length}`
                  : '0/0'}
              </span>
            </li>
            <li>
              <strong>ROI per Closed Action</strong>
              <span>{roi?.completed_action_plans ? money((roi.total_dollars_saved || 0) / roi.completed_action_plans) : '—'}</span>
            </li>
          </ul>
        </article>

        <article className="panel">
          <h3>Operator Guidance</h3>
          <ul className="guidance-list">
            <li>Prioritize `HIGH` risk cases with `escalated=true` for contract-level review.</li>
            <li>If coverage is below 30%, trigger more audit runs from recent transactions.</li>
            <li>Create action plans immediately for high-impact cases to move ROI dashboards.</li>
          </ul>
        </article>
      </div>

      <article className="panel">
        <h3>Activity Timeline</h3>
        {!liveSnapshot?.recent_events?.length ? (
          <p className="subtle">Waiting for live activity events…</p>
        ) : (
          <ul className="timeline-list">
            {liveSnapshot.recent_events.slice(0, 8).map((event: RealtimeEvent) => (
              <li key={event.id}>
                <div>
                  <strong>{event.title}</strong>
                  <p>{event.detail}</p>
                </div>
                <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
              </li>
            ))}
          </ul>
        )}
      </article>

      <article className="panel">
        <h3>Source-Level Metrics</h3>
        {sourceMetricsLoading ? <p>Loading source metrics…</p> : null}
        {!sourceMetrics?.length && !sourceMetricsLoading ? (
          <p className="subtle">No source metrics available yet. Run orchestration to populate scored coverage.</p>
        ) : null}
        {sourceMetrics?.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Agency Family</th>
                  <th>Category Family</th>
                  <th>Total</th>
                  <th>Scored</th>
                  <th>Scoring Coverage</th>
                  <th>Audit Cases</th>
                  <th>Escalated</th>
                  <th>High Risk</th>
                </tr>
              </thead>
              <tbody>
                {sourceMetrics.map((row) => (
                  <tr key={`${row.source_system}-${row.agency_family}-${row.category_family}`}>
                    <td>{row.source_system}</td>
                    <td>{row.agency_family}</td>
                    <td>{row.category_family}</td>
                    <td>{row.total_transactions}</td>
                    <td>{row.scored_transactions}</td>
                    <td>{percent(row.score_coverage_pct)}</td>
                    <td>{row.audit_cases}</td>
                    <td>{row.escalated_cases}</td>
                    <td>{row.high_risk_cases}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </article>
    </section>
  )
}

function CasesPage({ liveSnapshot }: { liveSnapshot: RealtimeSnapshot | null }) {
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [riskFilter, setRiskFilter] = useState<string>('')

  const { data, isLoading, error } = useQuery<{ items: CaseSummary[]; total: number }>({
    queryKey: ['cases', statusFilter, riskFilter],
    queryFn: () =>
      fetchCases({
        page: 1,
        page_size: 50,
        ...(statusFilter ? { status: statusFilter } : {}),
        ...(riskFilter ? { risk_level: riskFilter } : {}),
      }),
  })

  return (
    <section className="page">
      <header className="page-header">
        <h2>Audit Inbox</h2>
        <p>Filter and prioritize suspicious transactions before deep forensic actioning.</p>
      </header>

      <div className="toolbar">
        <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
          <option value="">All Status</option>
          <option value="OPEN">OPEN</option>
          <option value="IN_REVIEW">IN_REVIEW</option>
          <option value="CLOSED">CLOSED</option>
        </select>
        <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
          <option value="">All Risk</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </select>
      </div>

      {isLoading ? <p>Loading cases…</p> : null}
      {error ? <p className="error">Failed to load cases. Verify backend and database are running.</p> : null}

      {liveSnapshot?.recent_cases?.length ? (
        <article className="panel compact">
          <h3>Live Case Pulse</h3>
          <div className="pulse-row">
            {liveSnapshot.recent_cases.slice(0, 6).map((item) => (
              <span key={item.id} className={clsx('pulse-pill', (item.risk_level || 'LOW').toLowerCase())}>
                {item.risk_level || 'LOW'} · {item.status}
              </span>
            ))}
          </div>
        </article>
      ) : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>ML Score</th>
              <th>Risk</th>
              <th>Status</th>
              <th>Triage Escalated</th>
              <th>Impact</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {(data?.items ?? []).map((item: CaseSummary) => (
              <tr key={item.id}>
                <td>{item.id.slice(0, 8)}…</td>
                <td>{item.ml_score.toFixed(4)}</td>
                <td>{item.risk_level ?? '—'}</td>
                <td>{item.status}</td>
                <td>{item.groq_escalated ? 'Yes' : 'No'}</td>
                <td>{money(item.estimated_impact_usd)}</td>
                <td>
                  <Link to={`/cases/${item.id}`}>Inspect</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function CaseWorkspacePage() {
  const { caseId } = useParams<{ caseId: string }>()
  const queryClient = useQueryClient()
  const [createdPlanId, setCreatedPlanId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery<CaseDetail>({
    queryKey: ['case', caseId],
    queryFn: () => fetchCaseDetail(caseId ?? ''),
    enabled: Boolean(caseId),
  })

  const mutation = useMutation({
    mutationFn: createActionPlan,
    onSuccess: (response) => {
      const maybeId = (response as { id?: string })?.id
      setCreatedPlanId(maybeId ?? null)
      queryClient.invalidateQueries({ queryKey: ['roi'] })
      queryClient.invalidateQueries({ queryKey: ['coverage'] })
      queryClient.invalidateQueries({ queryKey: ['action-plans', caseId] })
    },
  })

  const { data: plansData } = useQuery<{ items: ActionPlan[]; total: number }>({
    queryKey: ['action-plans', caseId],
    queryFn: () => fetchActionPlans({ case_id: caseId ?? '', page: 1, page_size: 20 }),
    enabled: Boolean(caseId),
    refetchInterval: 30_000,
  })

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!caseId) return

    const formData = new FormData(event.currentTarget)
    mutation.mutate({
      caseId,
      payload: {
        owner_email: String(formData.get('owner_email') ?? ''),
        owner_department: String(formData.get('owner_department') ?? ''),
        action_type: String(formData.get('action_type') ?? 'ESCALATE') as ActionType,
        deadline: String(formData.get('deadline') ?? ''),
        estimated_recovery_usd: Number(formData.get('estimated_recovery_usd') ?? 0),
        notes: String(formData.get('notes') ?? ''),
      },
    })
  }

  if (isLoading) {
    return (
      <section className="page">
        <p>Loading case details…</p>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="page">
        <p className="error">Unable to load case workspace.</p>
      </section>
    )
  }

  const isEscalated = data.groq_escalated === true
  const triageDecision = data.groq_verdict?.reason ?? 'Triage pending. The orchestration pipeline has not completed triage for this case yet.'
  const deepAuditVerdict = isEscalated
    ? data.gemini_report?.verdict ?? data.llm_verdict ?? 'Escalated - deep audit pending'
    : 'Not applicable (no escalation)'
  const violatedClause = isEscalated
    ? data.violated_clause_id ?? 'Not identified yet'
    : 'Not applicable (no escalation)'
  const confidenceText = isEscalated
    ? typeof data.confidence === 'number'
      ? data.confidence.toFixed(2)
      : 'Pending'
    : 'Not applicable (no escalation)'
  const clauseCitation = isEscalated
    ? data.contract_clause_cited ?? 'Deep audit citation pending.'
    : 'No deep-audit clause citation generated because triage did not escalate this case.'

  return (
    <section className="page">
      <header className="page-header">
        <h2>Forensic Case Workspace</h2>
        <p>Evidence-driven review with immediate remediation assignment and ROI impact tracking.</p>
      </header>

      <div className="workspace-grid">
        <article className="panel">
          <h3>AI Evidence</h3>
          <p><strong>Case ID:</strong> {data.id}</p>
          <p><strong>ML Score:</strong> {data.ml_score.toFixed(4)}</p>
          <p><strong>Risk Level:</strong> {data.risk_level ?? '—'}</p>
          <p><strong>Triage Decision:</strong> {triageDecision}</p>
          <p><strong>Deep Audit Verdict:</strong> {deepAuditVerdict}</p>
          <p><strong>Violated Clause:</strong> {violatedClause}</p>
          <p><strong>Confidence:</strong> {confidenceText}</p>
          <div className="interpretation-box">
            <strong>System Interpretation</strong>
            {isEscalated ? (
              <p>The transaction was escalated for deep analysis. Prioritize legal and remediation actions.</p>
            ) : (
              <p>The triage stage judged current evidence as low risk. This is not a failure; you can still create preventive action plans or re-trigger audit if context changes.</p>
            )}
          </div>
          <pre className="evidence-block">{clauseCitation}</pre>
        </article>

        <article className="panel">
          <h3>Action Plan</h3>
          <form className="form" onSubmit={onSubmit}>
            <label>
              Owner Email
              <input type="email" name="owner_email" placeholder="legal.ops@agency.gov" required />
            </label>
            <label>
              Department
              <input type="text" name="owner_department" placeholder="Legal / Procurement / Finance" />
            </label>
            <label>
              Action Type
              <select name="action_type" defaultValue="ESCALATE">
                <option value="CLAWBACK">CLAWBACK</option>
                <option value="PAYMENT_HALT">PAYMENT_HALT</option>
                <option value="VENDOR_REVIEW">VENDOR_REVIEW</option>
                <option value="CONTRACT_RENEGOTIATION">CONTRACT_RENEGOTIATION</option>
                <option value="TEMPLATE_UPDATE">TEMPLATE_UPDATE</option>
                <option value="ESCALATE">ESCALATE</option>
                <option value="DISMISS">DISMISS</option>
              </select>
            </label>
            <label>
              Deadline
              <input type="datetime-local" name="deadline" required />
            </label>
            <label>
              Estimated Recovery (USD)
              <input type="number" name="estimated_recovery_usd" min="0" step="0.01" defaultValue="0" />
            </label>
            <label>
              Notes
              <textarea name="notes" rows={4} placeholder="Execution guidance, legal notes, and escalation details" />
            </label>
            <button type="submit" disabled={mutation.isPending}>
              {mutation.isPending ? 'Creating…' : 'Create Action Plan'}
            </button>
          </form>

          {mutation.isSuccess ? <p className="success">Action plan created and linked to this case.</p> : null}
          {mutation.isError ? <p className="error">Failed to create action plan. Validate required fields and try again.</p> : null}
          {createdPlanId ? <p className="subtle">Created Plan ID: {createdPlanId}</p> : null}
        </article>

        <article className="panel">
          <h3>Action Plan History</h3>
          {!plansData?.items?.length ? (
            <p className="subtle">No action plans yet for this case.</p>
          ) : (
            <ul className="timeline-list">
              {plansData.items.map((plan: ActionPlan) => (
                <li key={plan.id}>
                  <div>
                    <strong>{plan.action_type} · {plan.status}</strong>
                    <p>{plan.owner_email} · Recovery {money(plan.estimated_recovery_usd)}</p>
                  </div>
                  <time>{plan.created_at ? new Date(plan.created_at).toLocaleString() : '—'}</time>
                </li>
              ))}
            </ul>
          )}
        </article>
      </div>
    </section>
  )
}

function ContractsPage() {
  const [input, setInput] = useState('')
  const [query, setQuery] = useState('')

  const { data: contracts, isLoading } = useQuery<{ items: ContractItem[] }>({
    queryKey: ['contracts'],
    queryFn: () => fetchContracts({ page: 1, page_size: 50 }),
  })

  const { data: searchData, isFetching } = useQuery<{ query: string; results: Array<{ text: string; score: number }> }>({
    queryKey: ['contracts-search', query],
    queryFn: () => searchContracts({ q: query }),
    enabled: query.length > 2,
  })

  const onSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setQuery(input.trim())
  }

  return (
    <section className="page">
      <header className="page-header">
        <h2>Smart CLM Repository</h2>
        <p>Semantic contract search for clause-level compliance checks and procurement guidance.</p>
      </header>

      <form className="toolbar" onSubmit={onSearch}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder='Try: "split billing prohibition"'
        />
        <button type="submit">Search</button>
      </form>

      <div className="panel-grid">
        <article className="panel">
          <h3>Indexed Contracts</h3>
          {isLoading ? <p>Loading contracts…</p> : null}
          <ul className="list">
            {(contracts?.items ?? []).map((contract: ContractItem) => (
              <li key={contract.id}>
                <div>
                  <strong>{contract.title}</strong>
                  <p>{contract.id.slice(0, 20)}…</p>
                </div>
                <span>{contract.status}</span>
              </li>
            ))}
          </ul>
        </article>

        <article className="panel">
          <h3>Clause Search Results</h3>
          {isFetching ? <p>Searching…</p> : null}
          {!searchData?.results?.length ? <p className="subtle">Search to retrieve top clause matches.</p> : null}
          <ul className="list">
            {(searchData?.results ?? []).map((hit: { text: string; score: number }, idx: number) => (
              <li key={`${idx}-${hit.score}`}>
                <div>
                  <strong>Relevance: {hit.score.toFixed(3)}</strong>
                  <p>{hit.text.slice(0, 260)}...</p>
                </div>
              </li>
            ))}
          </ul>
        </article>
      </div>
    </section>
  )
}

function KpiCard({ title, value }: { title: string; value: string }) {
  return (
    <article className="kpi-card">
      <h3>{title}</h3>
      <p>{value}</p>
    </article>
  )
}

export default App
