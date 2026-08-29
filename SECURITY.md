# Security Policy

Please report suspected credential exposure, unsafe checkpoint handling, dependency vulnerabilities, or model-loading issues privately to the repository owner through GitHub's security advisory feature. Do not open a public issue containing secrets or exploit details.

Never load untrusted PyTorch checkpoints. Verify published SHA-256 hashes before use and prefer weights-only loading where supported.
