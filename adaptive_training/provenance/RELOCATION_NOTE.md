# Dependency relocation note

On 2026-11-08, exact copies of the frozen arrays, observational benchmarks,
reference-only summaries, parity receipts, author-equation source, and
intervention helpers required by this experiment were placed under
`data/source_inputs/`, `provenance/source_parity/`, and `vendor/`. Their hashes
are checked by `scripts/00b_verify_self_contained.py`.

Some immutable receipts in `provenance/`, `outputs/frozen/`, and
`outputs/evaluation/` retain absolute paths from the original execution. Those
strings document where a file was read when the experiment was frozen; they
are not live dependencies. Reruns use only the local paths defined by the
current scripts. The original hashes and numerical results were not changed by
the relocation.
