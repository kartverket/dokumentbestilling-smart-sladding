# Contributing to dokumentbestilling-smart-sladding

Built by Kartverket, published as open source under the MIT license.

## How to contribute

1. Open an issue describing the problem or the change before you start.
2. Fork the repo and branch off `main`. Keep commits small and focused.
3. Run the checks below.
4. Open a pull request against `main`.

## Checks

CI runs `pytest` from `app/` on every push and pull request. Two self-tests run
without a GPU, a server or access to any documents:

```sh
python utils/fnr_vakt.py --selftest    # the fnr guard catches what it should
python utils/vlm_selftest.py           # vlm_export -> vlm_judge -> vlm_evaluate
```

Do not commit sensitive data, test documents with real personal information, or
environment secrets.

## Fødselsnummer in code

The repo is public. Never use real fødselsnumre as example or test values in
source, prompts or fasit files, not even ones from an uttrekk you have lawful
access to. Use synthetic numbers from
[Tenor](https://skatteetaten.github.io/testnorge/) (month + 80), or numbers
with invalid check digits.

A pre-commit hook enforces this. Enable it once per clone:

```sh
git config core.hooksPath .githooks
```

`source activate.sh` does the same automatically. The hook runs
`utils/fnr_vakt.py --staged`, which looks only at the lines a commit adds and
stops numbers that both pass the mod-11 check and carry a valid date.
Coordinates, dagboknumre and ids pass.

```sh
python utils/fnr_vakt.py --all       # scan the whole working tree
```

If a kontonummer or another id is flagged by accident, write `fnr-ok` in a
comment on the same line.

## Security issues

Do not report security issues in public issues. See
[SECURITY.md](.github/SECURITY.md).
