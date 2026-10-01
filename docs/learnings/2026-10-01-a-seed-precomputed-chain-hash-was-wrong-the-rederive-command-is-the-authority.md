# A seed's precomputed ledger chain hash was wrong; its own re-derive command, run on the file as written, is the authority

ts: 2026-10-01T03:33:37Z
commit: 6377036
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("Holdover ARR design & experiments"), executing ~/dev/briefs/2026-09-30-seed-lag-ladder-0003-repin-instrument.md
status: refuted-assumption
fact: The 0003 re-pin seed stated the chain value for entry 0003 as `e4723cdc…` "with the ledger exactly as committed at e844494 and 0003 appended at end of file". Appending the seed's own entry text and running the seed's own re-derive command gives `b5e0ecf0…`, which `ledger_check` accepts; `e4723cdc…` would have been refused. The seed was right to say "re-derive it rather than trusting the number" — a chain value computed in another session, against a text that is then pasted with any difference in leading blank lines or trailing newline, does not survive. Same family as learning 2026-09-25 (chain cuts at the heading): the hash is a property of the bytes as written, never of a description of them.
basis: `python -c "import re,hashlib;t=open('ledger/ledger.md',encoding='utf-8').read();h=re.search(r'^## Entries\s*$',t,re.M).start();e=t.index('### 0003');print(hashlib.sha256(t[h:e].encode()).hexdigest()[:16])"` → `derived b5e0ecf0f2a72016`; `grep -o "e4723cdc…" ~/dev/briefs/2026-09-30-seed-lag-ladder-0003-repin-instrument.md` → `seed-stated e4723cdc6d69114e` (captured 2026-10-01T03:33:37Z at 6377036, the commit that carries 0003 with the derived value; `ledger_check` → `ledger ok (blocks unchanged vs HEAD)`).
re-verify: .venv/Scripts/python.exe -m lag_ladder.ledger_check
