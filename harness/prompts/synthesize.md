<!-- Rendered with str.format_map. Placeholders: {noun}, {label_set}, {evidence_json}, {hypotheses_json}, {step_titles}. -->
<!-- Optional placeholders this stage also accepts: {prompt}, {quantifier}, {audio_text}. -->
<!-- A literal brace must be written twice: {{ becomes { and }} becomes }. -->

Write the reasoning for one video clip using only the record below.

EVIDENCE RECORD

{evidence_json}

RANKED READINGS

{hypotheses_json}

The question asks for the {noun} displayed in the video.
The entries under consideration are: {label_set}

Write one short paragraph for each of these headings, in this order:

{step_titles}

Rules:

- Every sentence must cite at least one evidence id in square brackets, written as `[v2]` or `[a1, a3]`. A sentence with no citation is deleted before anyone reads the answer.
- Cite only ids that appear in `items`. An invented id is deleted the same way.
- Keep each paragraph to two or three sentences and do not repeat a heading inside its paragraph.
- The last paragraph must name the strongest alternative reading and say what speaks against it.
- Do not write a closing sentence and do not state the final answer; that sentence is written for you.

Return a single JSON object and nothing else, with these keys:

- `steps`: a list of exactly four strings, one paragraph per heading, in the order above.
- `justification`: one sentence, with citations, saying why the top-ranked reading beats the strongest alternative.

Output JSON only: no prose before or after it, no code fence, no comments.
