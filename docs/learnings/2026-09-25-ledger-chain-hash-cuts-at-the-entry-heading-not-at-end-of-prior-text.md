# The ledger chain hash cuts at the new entry's heading, not at the end of the prior text: a hand-computed hash over "everything so far" is one newline short

ts: 2026-09-25T04:27:24Z
commit: 04169c6abbb7fb01e1b0818192e8ef11442b6330
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("Holdover ARR design & experiments"), appending entry 0002 (refusal observed 2026-09-24, re-demonstrated at close)
status: refuted-assumption
fact: `ledger_check._check_chain` hashes `text[entries_start : <start of this entry's "### NNNN" heading>]`. When an entry is appended as "\n### 0002 …" after a file that ends in "\n", the blank separator line is inside the hashed span. Computing the hash over the pre-append text (up to its last byte) omits that newline and the checker refuses with "does not match the entries section above it". The correct recipe is: write the entry, then compute `chain_hash(text, heading.start(), entries_start)` on the file as it stands, and substitute. The gate caught the wrong hash before anything was committed; the committed 0001 block was never touched.
basis: first attempt refused: `LEDGER: entry 0002: prior-entries-sha256 e9a40963878d... does not match the entries section above it (51bd44723ec5...)` (captured 2026-09-24 on the working tree before commit 04169c6); re-demonstrated at close: `chain_hash(s, m.start(), head)` → `51bd44723ec5`, `chain_hash(s, m.start()-1, head)` → `e9a40963878d` (captured 2026-09-25T04:27:24Z at 04169c6); `python -m lag_ladder.ledger_check` → `ledger ok (blocks unchanged vs HEAD)`.
re-verify: .venv/Scripts/python.exe -m lag_ladder.ledger_check
