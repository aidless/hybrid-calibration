# Contributing

Thanks for your interest in improving this repository.

## Scope

This is a research artifact. Changes that affect **research claims, statistics, or
paper text** are not accepted as routine pull requests: they need to be discussed in
an issue first, because a claim change requires re-running the evidence pipeline and
regenerating the manifest, not just editing prose.

Contributions that fit the ordinary flow:

- bugs in the code or the verification scripts
- reproducibility problems (a recomputation that fails, a missing dependency, an
  unclear instruction)
- documentation that contradicts the code
- additional tests and negative-control fixtures
- translation and formatting

## Getting set up

```bash
git clone https://github.com/aidless/hybrid-calibration.git
cd hybrid-calibration
pip install -r requirements.txt
```

## Verifying your change

```bash
python3 -m compileall .
```

A pull request that does not state how it was verified will not be reviewed.

## Claim-changing changes

Open an issue that states, in order:

1. the current claim and where it is stated (file and field)
2. the new evidence, and how it was produced (script, seed, environment)
3. what changes in the manuscript, appendix, or evidence manifest as a result

Research changes are landed only after the evidence recomputes and the manifests
match. Expect the review to ask for the recomputation output.

## Commit messages

One logical change per commit. Reference the issue. No unrelated reformatting in the
same commit.
