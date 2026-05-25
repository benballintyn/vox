"""Live provider integration tests for vox.

Gated by the ``integration`` marker (declared in pyproject.toml). Tests
in this package hit real provider APIs and require the relevant API key
in the environment — they skip cleanly when a key is absent.

The point of this suite is to catch *drift in the boundary*: that the
shapes vox's translators assume still match what real providers return,
and that the requests vox emits are still accepted. Mocked tests prove
"given response shape X we produce Y"; only live tests prove X is still
the real shape.

Tests assert on STRUCTURE, never on model content (with a small number
of constrained-prompt smoke checks used as gross-breakage canaries).
"""
