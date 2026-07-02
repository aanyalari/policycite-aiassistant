# Cotiviti submission checklist

## Required artifacts

- [x] POC source code
- [x] Two-page report plus third-page bibliography (`.docx`)
- [ ] Project-author evaluation-label signoff
- [ ] Fresh benchmark after answer-cleanup guardrails
- [ ] PowerPoint deck (`.pptx`)
- [ ] Current PolicyCite screenshots
- [ ] On-camera presentation and POC demonstration (`.mp4`)

## Technical verification

- [x] Generation trace contains only passages supplied to the model
- [x] Abstentions contain no false generation-context IDs
- [x] Scorer rejects supported statements without contributing citations
- [x] Condition-specific coverage labels
- [x] Original and invalid runs preserved as audit history
- [x] Unit tests pass
- [x] Dependencies are pinned
- [x] GitHub Actions test workflow is present
- [ ] Fresh live `/ask_cited` smoke test after Ollama restart

## Author signoff

Before submission, open
`evaluation/results/20260701T231854Z-schema-v2-provenance-fixed.json` and review
every `human_labels` value using `evaluation/LABELING_GUIDE.md`. Replace the
artifact's `reviewer` and `project_author_signoff` fields with your name, review
date, and `complete`, then rerun the score command.

Because later answer-cleanup guardrails change generated outputs, run a fresh
timestamped evaluation when Ollama is available and use that artifact—not the
older metrics—in the final report and slides.

## GitHub and email

- [ ] Review `git diff` and ensure `.env` or credentials are absent
- [ ] Commit the final code and artifacts
- [ ] Push the nested `medical-rag-ai-assistant` repository
- [ ] Confirm the GitHub repository opens from a signed-out browser
- [ ] Upload the final MP4 if it is not already tracked
- [ ] Share the repository with `jesus.hurtado@cotiviti.com`
- [ ] Email the repository link with subject:
  `INTERN - Aanya Lari - <college or university>`
