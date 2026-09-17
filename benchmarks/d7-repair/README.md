# Pinned D7 numerical repair and compiled replay

These are the source-bound workers, kernels and adapters used for the completed
[compiled 10K M1/M8 experiment](https://github.com/Terrydaktal/qwen-r9700-lab/releases/tag/d7-report-v1.0.0).
This is an experimental qualification path. It does not change the container's
default inference configuration or automatically accept a different source build.

```text
d7-repair/
├── stock_gdn_*                # convolution, recurrence, causal prefill and runtime binding
├── stock_m1_*                 # serial arithmetic for normalization, attention and BF16 head
├── build_*                    # CPU-side HIP compilation and source/binary receipts
├── probe_*                    # small native correctness/state/graph/performance gates
├── optimized_d7_*             # admitted optimized paths and pre-capture installation
├── benchmark_*                # forced-token replay and separate unprofiled speed controls
├── trace_private_d7_rows.py   # owner-only aligned logit capture
├── tests/                    # CPU adapter, admission and replay regressions
├── configs/profiles/          # pinned finite-precision reference contract
└── SOURCE_PROVENANCE.json
```

Apply the companion conformance-support PR first. Its `qwen_r9700_lab` package
provides the shared evidence, state and process contracts.

```sh
uv run --project benchmarks/conformance --extra cpu-tests \
  pytest benchmarks/d7-repair/tests -q
```

One compiler-identity test additionally requires the preserved, hashed reference
HSACO and metadata; set `RADIANCE_GDN_REFERENCE_ARTIFACTS` to that artifact directory.
A skip is not native qualification. The other CPU tests use synthetic data and do
not access a GPU. No private Pi fixture or generated text is included here.

Native execution order:

1. Select the exact backend/model contract and source binding. Supply your own
   owner-only token fixture, native specification and isolated output directory.
2. Use `build_stock_m1_norm.py`, `build_stock_m1_head_pair.py`,
   `build_stock_m1_attention_shared.py` and `build_packed_gdn_transport.py` to create
   authenticated builds. Their inputs are explicit source/build directories;
   their outputs are shared libraries, compiler logs and JSON build receipts.
3. Run the matching `probe_*` entry points under the shared exclusive GPU lease.
   These compare outputs and persistent state, test graph replay and injected
   faults, and record the qualified build identities. Consult each script's
   `--help` for its admitted geometry and required native spec.
4. `qualify_stock_gdn_model.py` binds the numerical repair. The optimized workers
   bind the stage receipts before tracing/capture; unsupported shapes retain the
   repaired fallback. Never reseal a changed build as if old evidence covered it.
5. `benchmark_compiled_d7_corpus.py` consumes a sealed corpus and repair/performance
   manifests, checkpoints each continuation, audits actual compiled graph use and
   produces separate private aligned rows and public aggregate comparisons.
   `benchmark_optimized_d7.py` measures speed separately without correctness tracing.

The complete report contains top-1/10/20 set and ordering results, every compiled
GPU stage/group with old/fixed timing and its repair, and source/binary seals.
All-seven-accepted replay passed 10,000/10,000 positions and 23/23 prefills.
Eager-versus-compiled M1 remains a separately measured unresolved discrepancy.

The code intentionally rejects source or geometry drift. Porting these adapters
to newer dependencies needs fresh qualification; the published result is tied
to the pinned Radiance 1.0.16 / vLLM 0.28 environment. This PR provides the complete
experimental integration for review, while small core changes are submitted to
vLLM and libr4d separately.
