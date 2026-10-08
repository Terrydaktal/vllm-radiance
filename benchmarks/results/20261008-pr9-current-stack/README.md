# Current-stack verify-head contracts

Radiance 1.0.387 / vLLM 0.30 / PyTorch 2.12 / ROCm 7.14 on R9700.
Baseline PR checks pass. CPU dispatch tests: 117 passed, 19 native opt-in skips.
Native suite: 136 passed, zero skips on GPU 0.

This covers synthetic selection, dispatch, full-head fallbacks and a TP2 eligibility guard; it does not claim distributed TP2 numerical execution, complete candidate recall, serving throughput or end-to-end output qualification. Historical benchmark receipts remain separate.
