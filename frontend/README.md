# Frontend (React + Vite)

Operational UI for procurement audit intelligence.

## Views

- Executive Dashboard
- Audit Inbox
- Forensic Case Workspace
- Smart CLM Search

## Runtime config

Create `.env.local`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

If `VITE_API_BASE_URL` is omitted, the app still defaults to `http://127.0.0.1:8000` for websocket/API routing.

## Scripts

```bash
npm install
npm run dev
npm run build
npm run preview
```

## Notes

- Uses React Query for cache + stale data control.
- Uses websocket stream `/api/v1/realtime/stream` for near-realtime KPI/event updates.
- UI language is provider-agnostic; it focuses on workflow semantics.
