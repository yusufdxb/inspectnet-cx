# Changelog

## v0.1.0-inspectnetcx-hf-release - pending Hub publish

- Added compact Hugging Face package with claims ledger, artifact index, verified reports, prediction examples, and release visual.
- Added clean-venv install path via `requirements.txt`.
- Added tested environment, CUDA boundary, and disk planning notes.
- Sanitized public HF package artifacts to avoid raw dataset files, checkpoints, absolute local paths, caches, and secrets.
- Clean outside-repo validation passed on Python 3.10.12.
- Hugging Face upload is blocked until `hf` CLI has a valid user access token.

## 0.0.1

- Added Phase 0 package scaffold.
- Added HF-style config, processor, model, and output dataclass.
- Added local inference CLI.
- Added baseline placeholder harness and aggregation.
- Added proof-readiness and latency tools.
- Added release documentation and model card.
