---
status: accepted
date: 2026-10-16
decision-makers: Zack Arno
---

# Keep the scheduled download workflow alive by re-enabling it via the API

## Context and Problem Statement

`Download Latest WHO PDF` is the pipeline's only scheduled workflow. Both
extraction workflows trigger on its completion (`workflow_run`), and
post-processing is called from those. GitHub disables scheduled workflows in
public repositories after 60 days without repository activity. Commits to this
repo are sporadic: after the last push on 2026-04-13 the workflow was disabled
on 2026-06-16, and the whole pipeline stopped for four months with no alert.
How do we keep the schedule running when nobody commits?

## Considered Options

* Re-enable the workflow via the API (`PUT /actions/workflows/{id}/enable`) from a schedule-only job in the workflow itself
* The same call via the third-party `liskin/gh-workflow-keepalive` action
* Commit to the repo from the workflow so there is always recent activity
* Trigger runs from an external scheduler (`gh workflow run` / `repository_dispatch`)
* Make the repository private (the rule only applies to public repos)
* Re-enable manually when someone notices

## Decision Outcome

Chosen option: "Re-enable via the API from a schedule-only job", because it is
one API call with no commits, no third-party code, and no extra infrastructure.

### Consequences

* Good, because history stays free of bot commits and no external action is
  given an `actions: write` token.
* Bad, because it relies on behavior GitHub does not document: the docs talk
  about "repository activity", and that an API re-enable resets the 60-day
  clock is established only by practice (`liskin/gh-workflow-keepalive`,
  `keepalive-workflow` v2). If that changes, the workflow is disabled again
  about 60 days after the last commit; `gh workflow list` shows the state.
* Bad, because the job only runs on a schedule, so it cannot revive a workflow
  that is already disabled; that still takes a manual `gh workflow enable`.
* Neutral: if the keepalive job fails, the whole Download run fails and that
  run's extraction is skipped. The next run catches up, since extraction picks
  the most recently uploaded PDF.

Rejected options:

* `liskin/gh-workflow-keepalive`: the same single `gh api` call, plus a
  third-party dependency holding a write-scoped token.
* Commits from the workflow: GitHub documents these as activity, but they add
  two bot commits a week to history and need `contents: write`.
* External scheduler: another system and credential to run for a one-line problem.
* Private repo: changes the visibility of an open repo to work around an ops limit.
* Manual re-enable: this is what happened in June 2026, and nobody noticed for four months.
