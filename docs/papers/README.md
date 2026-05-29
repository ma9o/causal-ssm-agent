# Reference papers

Local-only PDFs of reference papers. The PDFs are **gitignored** — only
[`papers.txt`](papers.txt) (the arxiv manifest) is tracked, so the set of
papers is reproducible without bloating the repo.

## Usage

```bash
# Download every paper in the manifest that isn't already here
scripts/fetch_papers.sh

# Add a paper by arxiv id (slug optional — defaults to the id) and download it
scripts/fetch_papers.sh 1906.02691 vae-intro
```

The arxiv id may be bare (`2301.12345`), versioned (`2301.12345v2`), or a full
`arxiv.org/abs/...` / `arxiv.org/pdf/...` URL.
