---
status: accepted
date: 2026-11-21
decision-makers: Zack Arno
---

# Use Azure OpenAI for LLM extraction instead of the direct OpenAI API

## Context and Problem Statement

The LLM extraction workflow called `api.openai.com` with a project key
(`DSCI_AZ_OPENAI_API_KEY_WHO_CHOLERA`) billed to a prepaid OpenAI account. That
account ran out of credits, so every call failed with
`429 insufficient_quota`; the last successful LLM extraction was 2026-06-12.
The team already pays for an Azure OpenAI resource with a `gpt-5` deployment,
which hdx-signals uses. Where should this pipeline's OpenAI-family calls go?

## Considered Options

* Azure OpenAI, through its OpenAI-compatible `/openai/v1/` route
* Azure OpenAI, through the `AzureOpenAI` client class with an API version
* Top up the direct OpenAI account and change nothing
* Keep both, selected by an environment variable
* Route OpenAI models through OpenRouter, which the client already supports

## Decision Outcome

Chosen option: "Azure OpenAI through the `/openai/v1/` route", because it uses
billing the team already has, and the existing `OpenAI` client, Responses API
and streaming code work unchanged with only a `base_url`.

`LLMClient` now sends every OpenAI-family model to Azure and raises if
`AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_ENDPOINT` is missing. There is no
fallback to the direct OpenAI key.

### Consequences

* Good, because spend is on the organisation's Azure subscription rather than
  a prepaid balance that can silently reach zero.
* Good, because the variable names match hdx-signals, and the endpoint may be
  stored either as the bare resource URL or as a full deployment URL.
* Bad, because the `model` argument is now an Azure *deployment name*. A model
  works only if a deployment with that exact name exists on the resource
  (currently `gpt-5` and `gpt-4o`).
* Bad, because Azure rejects PDF uploads with purpose `user_data` (HTTP 500);
  uploads use purpose `assistants` instead.
* Neutral: the deployed `gpt-5` is the `2025-08-07` snapshot. Azure controls
  when a deployment's model version changes, so extraction output can shift
  without a change in this repo.
* Neutral: logs still record `provider: openai`. Runs from before and after
  the switch are distinguished by date, not by that field.
* Neutral: modules outside `LLMClient` that call OpenAI directly
  (`llm_extract.py`, `llm_text_extract.py`, `main.py`, narrative linking) were
  not migrated. They are not part of the scheduled pipeline.

Rejected options:

* `AzureOpenAI` client class: needs an `api-version` that must be recent
  enough for the Responses API, a third variable to keep current for no gain.
* Top up the OpenAI account: restores service but keeps a separate prepaid
  balance that failed without any alert.
* Keep both behind a switch: more configuration and a second code path for a
  provider with no credits.
* OpenRouter: another vendor and key for models the organisation already has
  on Azure.
