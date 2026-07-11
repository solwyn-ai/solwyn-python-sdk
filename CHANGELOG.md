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

---

Non-text modality support — SDK merge 2. Wire-contract changes are API-first:
Solwyn Cloud accepts every field below before this SDK releases.

### Added

- **Image generation is now budget-tracked.** `client.images.generate` and
  `client.images.edit` (native OpenAI and every OpenAI-compatible profile,
  including Together) are intercepted through the media-call lifecycle, emitting
  `modality="image"`. Native gpt-image reports token usage carrying per-modality
  buckets (`image_input_tokens` / `image_output_tokens`, documented subsets of
  input/output); compat, dall-e, and FLUX endpoints report no usage, so the
  billable basis is the request-derived per-image `MediaUsage`
  (`n` → `image_count`, `size` → `resolution`, `quality` → `quality`). Both bases
  ride the confirm when observable and the server's pricing-card unit picks
  (token card vs per-image card). The operation (generate vs edit) rides a
  private marker that is stripped before the provider call; the customer
  `prompt=` / `image=` / `mask=` bytes are never read.
- **Google `generate_images` (imagen) is now budget-tracked.**
  `client.models.generate_images` is intercepted through the same lifecycle,
  emitting `modality="image"`. imagen exposes no usage, so the request-derived
  per-image quantity is the sole billable basis: `config.number_of_images`
  (read duck-typed from both the config-object and dict shapes, defaulting to 1
  per the API contract) → `image_count`. An exact count, never a silent $0.
- **Google `usageMetadata` per-modality token buckets.** The Google chat
  extractor now maps the `prompt_tokens_details` (input side) and
  `candidates_tokens_details` (output side) `ModalityTokenCount` lists: IMAGE
  buckets → `image_input_tokens` / `image_output_tokens`; AUDIO buckets →
  `audio_input_tokens` / `audio_output_tokens`. This lets a
  `generate_content` call to a token-billed image model (e.g. gemini-3-pro-image)
  carry its image-output tokens so the server prices them at the image rate.
  Duck-typed with None-safety throughout; a response without per-modality details
  leaves every field at 0 exactly as before.
- **`MediaUsage` wire type + `estimated_media` + image token keys (wire window
  2).** `MediaUsage` — the non-token billable quantities and variant selectors a
  per-unit-priced surface bills on (`image_count`, `generation_count`,
  `video_seconds`, `audio_seconds`, `input_characters`, `resolution`, `quality`)
  — is vendored inline (the SDK is self-contained) and carried on
  `MetadataEvent.media_usage` and `BudgetConfirmRequest.media_usage`;
  `BudgetCheckRequest.estimated_media` carries the request-derived pre-flight
  quantity so the budget check prices a precise per-unit cost. `TokenDetails`
  gains `image_input_tokens` / `image_output_tokens`. Every quantity is `None`
  (never a zero-as-default) when unobservable, routing a known-unit card with a
  missing quantity to the server's unpriced lane rather than a real $0.

### Changed

- **`images` and Google `generate_images` graduated from warn-once to
  intercepted.** Both leave the unshipped-spend-surface warn-once set — a metered
  surface must not advertise itself as untracked (superseding the merge-1
  warn-once posture entry for those two surfaces). Remaining warn-once
  pass-throughs: OpenAI-dialect `audio` / `videos` and Google
  `generate_videos`.

---

Non-text modality support — SDK merge 3. Audio surfaces reuse the wire contract
shipped in merge 2 (`modality="audio"`, `MediaUsage.audio_seconds` /
`input_characters`, `TokenDetails.audio_input_tokens`) — no new wire fields.

### Added

- **Audio transcription is now budget-tracked.**
  `client.audio.transcriptions.create` (native OpenAI and every
  OpenAI-compatible profile, including Groq whisper) is intercepted through the
  media-call lifecycle, emitting `modality="audio"`. One extractor discriminates
  on `usage.type`: the token-billed models (the gpt-4o-transcribe family) settle
  on their audio-input token bucket (`audio_input_tokens`), while whisper reports
  an integer whole-second count (`usage.seconds`, present on any JSON
  `response_format`) that rides `MediaUsage.audio_seconds` — the fractional
  top-level `duration` field is never read. A non-JSON `response_format`
  (`text` / `srt` / `vtt`) returns a plain string with no usage, so the call is
  tracked UNPRICED with a one-time hint to pass a JSON `response_format` for
  priced tracking. The audio file bytes are never read.
- **Text-to-speech is now budget-tracked.** `client.audio.speech.create` (native
  OpenAI and every OpenAI-compatible profile) is intercepted through the same
  lifecycle, emitting `modality="audio"`. TTS responses carry no usage, so the
  sole billable basis is the request's `input` text length, measured in the
  privacy firewall as an exact character count (`MediaUsage.input_characters`)
  that rides both the pre-flight budget check and the settled confirm; the input
  text itself is never read, logged, or transmitted (tts-1 / tts-1-hd price from
  this character count server-side). Token-billed TTS models (`gpt-4o-mini-tts`
  and its dated snapshots) publish no usage of any kind — their audio-output
  tokens are unobservable, so by product decision those calls are warned once per
  process and passed through UNTRACKED rather than settled at an estimated or
  silent $0.

### Changed

- **`audio` graduated from warn-once to intercepted.** The `audio` attribute now
  returns intercepting machinery — its `transcriptions` and `speech` sub-surfaces
  are metered — leaving the unshipped-spend-surface warn-once set (a metered
  surface must not advertise itself as untracked, superseding the merge-1
  warn-once posture entry for `audio`). Its `translations` sub-surface stays
  recognized-but-untracked and warns once per process. Remaining warn-once
  pass-throughs: OpenAI-dialect `videos` and Google `generate_videos`.
