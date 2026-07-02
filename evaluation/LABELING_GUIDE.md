# Evaluation labeling guide

Use the frozen reference answer, required claims, and gold evidence in each
result. Do not copy the automated attribution verdict into a review field.

For each condition:

1. Mark a required claim `retained=true` only when the answer states it without
   a material contradiction.
2. Mark a statement `fully_supported=true` only when that condition's attached
   citations, considered together, support every material date, quantity,
   exclusion, modality, scope, and qualifier.
3. Mark a citation `contributes_support=true` when it provides full or partial
   support for its linked statement. Related-but-non-supporting passages are
   `false`.
4. A statement with no attached citations must be `fully_supported=false`.
5. Record ambiguities in `notes`; do not resolve them from outside knowledge.

The scorer enforces rule 4. Before submission, the project author must inspect
all ten items, record their name and review date in the artifact's `labeling`
object, and rerun the score command.
