# Manuscript House Style

This guide records the writing and revision style developed for this paper. Apply it to all
new manuscript text and all text added in response to reviewers. Do not rewrite established
text solely for style unless the author asks for a broader language pass.

## Core Principle

Use simple, plain, direct English. Technical ideas should remain technically exact, but the
reader should not have to decode the prose before understanding the science.

Prefer concrete verbs and nouns over abstract or pseudo-technical phrasing.

- Write “the inverse temperature affects retrieval and direct sampling in different ways,”
  not “the inverse temperature affects two related but distinct objects.”
- Write “we compared four sampling procedures,” not “we constructed a four-rung sampler
  ladder.”
- Write “the samples lay farther from the curve,” not “off-manifold error increased,” unless
  the formal term is needed and immediately defined.

## Build the Scientific Story

Every analysis should guide the reader through five questions:

1. Why did we perform the analysis?
2. What specific question or hypothesis did it test?
3. What did we compare or calculate?
4. What happened?
5. What can and cannot be concluded?

Do not introduce a metric without explaining why it answers the question. Do not report a
result before giving the reader enough context to interpret it. Add a short transition when
the text moves to a new biological or technical question.

## Introduction

- Develop the Introduction from general to specific: the broader scientific or practical
  problem, existing approaches and the particular unresolved need, then what this study did.
- Reserve the study-specific contribution for a developed final paragraph beginning
  “In this study, ...”. Do not announce the implementation before establishing the problem.
- Explain how the literature leads to the design question. Avoid a catalogue of methods
  followed by a disconnected list of experiments.

## Results Section Structure

- Do not use subsubsections in the Results. A ticker, model, case study, or individual
  figure is never a reason to create a numbered subsubsection.
- Default to one continuous Results narrative under the section heading. Use transitions
  in the prose to move between questions, comparisons, and case studies.
- Use a subsection only when the Results are unusually dense and the division is necessary
  for a reader to follow genuinely distinct analyses. Before adding one, first try a clear
  opening sentence, a transition paragraph, or the natural separation created by a table or
  figure.
- Do not use headings as substitutes for narrative glue. The sentence after a transition
  must explain why the next analysis follows from the preceding result.

## Sentences and Terminology

- Keep sentences direct and vary their length naturally. Connect related ideas with clear
  causal or explanatory links. Avoid both comma-linked chains of independent clauses and
  a succession of short declarations with the same rhythm.
- Do not begin a sentence with an acronym. For example, use “Stochastic attention
  nevertheless...” rather than “SA nevertheless....”
- Do not use em dashes. Use a comma, colon, parentheses, or a new sentence.
- Use technical terms only when they add precision. Define them at first use in language a
  biomedical reader can follow.
- Avoid inflated transitions and claims such as “critically,” “remarkably,” “intrinsic,”
  “machinery,” “harness,” or “substantive answer” when a direct statement will do.
- Avoid metaphors for technical workflows unless they make the method easier to understand.
- Define all unfamiliar metrics on first use, including what higher or lower values mean.
- Prefer “test,” “analysis,” or “comparison” over “attack” for membership inference unless
  discussing the formal literature term.
- Use “profiles” or “patients” consistently and do not call generated profiles independent
  clinical observations.

## Equations and Mathematical Transitions

- Introduce every displayed equation with a complete sentence that says what the equation
  gives, defines, or shows, and end the sentence with a colon.
- Do not lead into a displayed equation with a dangling phrase or comma. For example, write
  “Completing the square in each exponent gives the expression:” before the equation.
- When displayed equations occur in sequence, connect them with a short sentence that
  explains how the second follows from the first.

## Figures and Tables

- Begin every main and supplementary figure caption with a concise descriptive
  lead naming the comparison or quantity shown, such as “In-sample comparison
  of the implied volatility (IV) observed ...” or “Simulated evolution of ...”.
  Do not open with “The panels show,” “The figure shows,” or similar framing.
  A descriptive phrase is appropriate; a complete grammatical sentence is not
  required. Follow the lead with panel definitions, units, statistical keys,
  and other details needed to interpret the figure.

- Use the figures in `/Users/jdv27/Desktop/julia_work/HMM-w-jumps-paper` as the
  visual reference for manuscript figures. Follow their boxed panels, sans-serif
  labels, panel letters, inset legends, and navy/red palette. The author
  rejected the previous minimalist and serif redesigns. Keep background gridlines
  disabled, as explicitly requested, and inspect the figures at manuscript size.
- Put explanatory headlines, interpretation, and prose callouts in the caption,
  not on the plot. Allow only panel letters, legends, axis labels, and ticks within
  the graphic. Put ticker names, maturities, dates, and error summaries in the
  caption or a table, never in descriptive panel titles or annotation boxes.
- Cite every main-text figure and table in the Results at the point where its evidence is
  first reported or interpreted. A reference in the Methods, caption, or Discussion does not
  replace the Results citation.
- Place the figure or table reference in the sentence that states the corresponding finding.
  Do not use a detached “see Figure” or “see Table” sentence.
- Use `Fig.~\ref{fig:label}` for one figure, `Figs.~\ref{fig:first}--\ref{fig:last}` for
  multiple figures, and `Table~\ref{tab:label}` for a table. Use “Supplementary Fig.” and
  “Supplementary Table” for supplementary items.
- Identify the relevant panel inline when a claim depends on one panel, for example
  `(Fig.~\ref{fig:label}A)`.
- Build each Results passage in this order: explain the question, state what was compared or
  calculated, report the finding with its inline figure or table reference, and interpret what
  the finding does and does not show.
- Before considering the manuscript complete, inventory every `fig:`, `sfig:`, `tab:`, and
  `stab:` label and confirm that it is cited in the appropriate Results passage. Flag orphaned
  floats, Results claims without supporting references, and figures or tables cited only in
  Methods, captions, or Discussion.

## Project-Specific Technical Language

- Stochastic attention was not fit to the cohort. Patient profiles were stored as columns of
  the memory matrix used during sampling. Use “constructed the memory matrix,” “stored,” or
  “used during sampling.”
- Baseline models may be described as fit or estimated when that is what was done.
- Distinguish the Hopfield retrieval update from the corresponding sampling distribution.
  For the unit-normalized memories used here, multiplicity weights set the component
  probabilities and inverse temperature sets the spread around the selected profile.
- Generating 100 subgroup profiles does not create 100 independent patients or add clinical
  evidence beyond the source cohort.
- The source data were de-identified. The membership-inference test asked whether an
  already-known de-identified profile had been included in the memory matrix. It did not
  identify a person or reconstruct an unknown patient.
- Distinguish what was demonstrated from what was not tested. Use “was consistent with” when
  the analysis did not establish causation.
- Do not present future work as a promise. State the current limitation and, when useful, the
  type of method that could address it.

## Tense

- Results describing completed analyses and observed findings must be in the past tense.
- Methods describing what was done should normally be in the past tense.
- Mathematical definitions and general properties may use the present tense.
- Discussion statements about the study’s findings should remain appropriately bounded and
  should not shift into stronger present-tense claims than the results support.

## Paragraphs and Flow

- The author explicitly rejected the 109–148-word narrative paragraphs as too short and
  choppy. Do not use the former 100–200-word guideline or a word-count target as a measure
  of success. Prefer substantial paragraphs that develop a complete argument.
- Carry the motivation, comparison, evidence, and interpretation together when they belong
  to the same argument. A change of ticker, metric, or intermediate result does not by
  itself justify a paragraph break.
- Rewrite connections when combining related material. Simply removing paragraph breaks
  or padding sentences does not repair a choppy narrative.
- Break a paragraph when the central question or argument changes. Keep necessary short
  mathematical introductions and captions distinct from the narrative.
- Each paragraph should have a clear opening and build toward an inference or transition.
  Judge the result by continuity and the author's voice, not a uniform paragraph length.
- Preserve necessary detail, but move implementation detail to Methods or the Supplement
  when it interrupts the main narrative.
- End the Discussion with its limitations paragraph. Place computational scaling and other
  methodological scope considerations before that final paragraph.

## Discussion

- Do not summarize the Results paragraph by paragraph. Open with the main interpretive
  contribution and explain why it matters.
- Distinguish empirical findings from software demonstrations, accounting identities, and
  assumptions imposed by the model.
- Explain the tradeoff created by the proposed approach, including what becomes easier to
  diagnose and what structural coherence is lost.
- Use the Results to support synthesis. Do not repeat a sequence of numerical values when a
  table or figure reference and a direct interpretation are sufficient.
- Organize the Discussion around contribution, interpretation, practical implications, and
  evidentiary limits. Avoid a stand-alone laundry list of possible extensions.
- End with one consolidated limitations paragraph that states exactly which conclusions the
  study does not support.

## Claims and Limitations

- State the strongest claim supported by the analysis, but no stronger.
- A non-significant test does not establish equivalence.
- Reproducing a small training cohort is not the same as recovering its population.
- Mechanistic agreement for one modeled system does not validate unmodeled biological
  features.
- A generated point lying outside a convex hull does not by itself show implausible
  extrapolation in a sparse, high-dimensional cohort.
- Synthetic generation without a formal privacy mechanism is not anonymization and does not
  provide differential privacy.

## Reviewer Revisions

- Preserve the reviewer color macros: `\rone{...}`, `\rtwo{...}`, and `\rboth{...}`.
- Change only reviewer-added text unless the author explicitly approves edits to legacy
  prose.
- Keep the response-to-reviewers document aligned with the manuscript. It should explain
  what changed using the same interpretation and terminology as the paper.
- A response should answer the reviewer directly, then summarize the evidence and identify
  the manuscript location.

## Repository Workflow

For every manuscript revision:

1. Make the same textual change in `paper/` and `arxiv/`.
2. Update `peer-review-feedback/response-to-reviewers.md` when the change responds to a
   reviewer.
3. Preserve unrelated author edits.
4. Compile the manuscript and supplement with `make all` from `paper/`.
5. Run `git diff --check`.
6. Check the LaTeX logs for undefined references, overfull text, and other warnings.
7. Inspect the rendered passage when a change could affect layout or paragraph flow.

## Final Read-Aloud Test

Before considering revised prose complete, ask:

- Would a biomedical reader understand why this analysis was performed?
- Is every technical term necessary and explained?
- Does each sentence say exactly what happened?
- Is any claim stronger than the evidence?
- Could the same idea be stated more simply without losing precision?

If the answer to the last question is yes, simplify it.
