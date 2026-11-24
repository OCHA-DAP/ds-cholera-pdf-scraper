---
status: proposed
date: 2026-11-24
decision-makers: Zack Arno
---

# Pair repeated country/event rows one-to-one when comparing extractions

## Context and Problem Statement

The LLM and rule-based extractions are compared by matching rows on country
and event. That key is not unique: from week 30 of 2026 the bulletin lists two
"Complex Humanitarian crisis" rows for Chad, Ethiopia, the Central African
Republic and South Sudan, and two Mpox rows for Ghana. Merging on the key
compared every LLM row with every rule-based row for it, so two rows that both
extractions read correctly produced two false discrepancies. Weeks 30 to 41
reported 14 to 24 value discrepancies each; five were real across all twelve
weeks. How should rows that share a key be matched?

## Considered Options

* Pair rows one-to-one within a key, closest rows first
* Add the notification date to the key
* Number the repeated rows in the order they appear
* Keep the many-to-many merge and filter its output afterwards

## Decision Outcome

Chosen option: "Pair rows one-to-one within a key, closest rows first", because
it is the only option that handles both kinds of repetition seen in the data
(distinct rows with the same name, and a row one extraction emitted twice)
without depending on the two extractions agreeing on any single column.

Within a key, every LLM/rule-based combination is scored by the number of
fields that differ (the five compared fields plus the notification and
reporting-period dates), and pairs are taken from the lowest score up. A row
left without a partner is reported as present in only one extraction.

The key itself is now built from lower-case ASCII letters and digits, so
"Côte d'Ivoire" and "Cote d'Ivoire" are one record.

### Consequences

* Good, because a reported discrepancy now means two rows that are the same
  record disagree.
* Good, because a row duplicated by one extraction is now visible as an extra
  record. Before, an exact duplicate matched its twin and was not reported.
* Bad, because pairing picks the closest rows, which is the reading most
  favourable to agreement. If both extractions misread two repeated rows badly
  enough, the pairing could swap them and the reported differences would
  describe the wrong pair.
* Bad, because pairs are taken greedily rather than by an optimal assignment.
  The two can differ for three or more repeated rows that all disagree; the
  groups seen so far have two.
* Neutral: `comparison_key` in the unique-records file is now the reduced
  form (`cotedivoire_yellowfever`), not the display names.
* Neutral: event names that differ by more than spelling still do not match
  ("Complex Humanitarian crisis - ETH" against "Complex Humanitarian crisis").

Rejected options:

* Notification date in the key: pairs the week 30 to 41 rows, but fails where
  the date is identical (week 8 has the same Zimbabwe cholera row twice in
  both extractions) and turns any disagreement about the date into two unique
  records instead of one matched record.
* Order of appearance: the extractions do not return repeated rows in the same
  order, so this pairs the wrong rows.
* Filter the merge output: needs the same "which rows belong together" rule,
  applied later and to less information.
