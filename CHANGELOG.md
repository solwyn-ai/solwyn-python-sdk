# Changelog

All notable changes to the Solwyn Python SDK are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are
derived from git tags (hatch-vcs).

## [Unreleased]

Non-text modality support — SDK merge 1. Wire-contract changes are API-first:
Solwyn Cloud accepts every field below before this SDK releases.

### Added

- **Embeddings are now budget-tracked.** `client.embeddings.create` (native
  OpenAI and every OpenAI-compatible profile — xAI, DeepSeek, Mistral, Qwen,
  Groq, Together, Fireworks, Perplexity, Azure OpenAI, OpenRouter, Z.ai, Ollama,
  vLLM, LM Studio, and the generic catch-all) and Google's
  `client.models.embed_content` are intercepted through a lean media-call
  lifecycle: estimate → budget check → primary-only provider call →
  extract/measure → confirm + report. Embeddings emit `modality="embedding"` on
  the budget check, the confirm, and the metadata event. There is no candidate
  walk, no cross-provider translation, and no model fallback for media surfaces
  (embedding vectors are not interchangeable across providers).
- **`modality` wire field.** `MetadataEvent`, `BudgetCheckRequest`, and
  `BudgetConfirmRequest` each carry `modality`
  (`Literal["text","image","audio","video","embedding"]`, default `"text"`).
  Chat calls ride the `"text"` default and are behaviorally unchanged; media
  surfaces set the surface's modality. The server's pricing-card unit — not this
  label alone — selects the billing basis (core bills text tokens until
  per-modality cards land).

### Changed

- **Estimation fallback extended to embeddings.** When an embeddings response
  reports no usable usage block, billable input is derived from a length-based
  estimate off `input=` (str / list[str] via provider char-ratio; list[int] /
  list[list[int]] counted directly as pre-tokenized ids), flagged
  `token_details.is_estimated = true`, never a silent zero. Google's
  `EmbedContentResponse` currently exposes no `usage_metadata`, so Google
  embeddings are estimator-driven in practice; the usage-extraction path stays
  forward-compatible if Google adds it. An unobservable quantity settles as
  `None` — never a real $0 price.
- **Warn-once pass-through posture for unshipped spend surfaces.** Recognized but
  not-yet-intercepted billable surfaces (OpenAI-dialect `images` / `audio` /
  `videos`; Google `generate_images` / `generate_videos`) warn exactly once per
  process, then pass through untracked. The latch is per-process (not
  per-instance), so the create-a-client-per-request pattern stays quiet. Truly
  unrelated resources (files, moderations, batches, …) pass through silently.

### Fixed

- **Bedrock `start_async_invoke` now fails loud.** It joins `invoke_model` /
  `invoke_model_with_response_stream` in raising `ConfigurationError` instead of
  bypassing budget tracking — a video-scale spend surface whose usage lands
  out-of-band in S3 was a silent budget hole. Use Converse, or call the
  unwrapped client for deliberately untracked calls.
