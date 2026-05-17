# Security Policy

InspectNet-CX Phase 0 is not suitable for production inspection or safety-critical reject
decisions.

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.0.x | Yes |

## Reporting

Report security issues privately to the repository owner. Do not publish exploit details
before a fix or mitigation is available.

## Known Security Boundaries

- Loading arbitrary Hugging Face repositories with `trust_remote_code=True` can execute code.
- Phase 0 does not validate dataset provenance.
- Phase 0 does not provide supply-chain pinning beyond package version ranges.
- Phase 0 outputs must not be used as production reject decisions.
