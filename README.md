# TrendX

Autonomous trend research and analysis agent focused on agentic AI and LLM infrastructure.

## Quickstart

1. Install dependencies
2. Configure `configs/default.yaml`
3. Run the pipeline:

```bash
trendx run --config configs/default.yaml
```

## Commands

- `trendx ingest`
- `trendx analyze`
- `trendx report`
- `trendx run`
- `trendx backfill --weeks N`

## Notes

Outputs are written to `reports/YYYY-WW/brief.md` and `reports/YYYY-WW/brief.json`.
