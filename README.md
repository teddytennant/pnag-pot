# pnag-pot

Proximal Nesterov accelerated gradient for quadratically or elastic-net regularized partial optimal transport. Marginal inequalities are handled by an exterior penalty; the returned plan is projected onto the partial transport set by ROUND-POT.

```bash
pip install -e .
pytest
```

No color transfer, domain adaptation, or point-cloud benchmarks. No claimed speedups. Problem size n stays small.
