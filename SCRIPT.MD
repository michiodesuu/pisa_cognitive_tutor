# PISA Cognitive Tutor v2.0 — Skeptical Presentation Script

> **Format:** Solo presenter playing both advocate and skeptic. Skeptic challenges (in italics) appear before each answer. Use this to prepare for tough Q&A or to rehearse a self-critical walkthrough.

---

## Slide 1 — Opening: What Are We Actually Claiming?

Good morning. I want to walk you through the PISA Cognitive Tutor — a research platform we built for NECTEC to study how students engage cognitively when interacting with an AI tutor on PISA science questions.

But before I sell you on it, let me be honest about what is hard to believe, and why we built the system the way we did.

---

## Slide 2 — The Problem We're Solving

The central motivation is this: "active learning" is a label researchers and teachers throw around, but it means almost nothing scientifically. Is a student who highlights text "active"? What about one who copies notes? What about one who explains a concept back in their own words?

These behaviours are cognitively very different — but most studies collapse them into one category, measure them with self-report surveys, or proxy them through attendance records.

*Skeptic: So you're claiming you can do better than decades of educational research with an LLM chatbot and some Python scripts?*

No. We're claiming we can operationalise a specific theoretical framework — the ICAP Framework (Chi & Wylie, 2014) — in a way that is automated, scalable, and less subject to researcher bias than manual annotation. That is a narrower, more defensible claim.

---

## Slide 3 — The ICAP Framework

The ICAP Framework classifies cognitive engagement into four levels:

- **Passive** — student reads or listens, stores information
- **Active** — student selects, highlights, or repeats
- **Constructive** — student explains, infers, draws novel connections
- **Interactive** — student debates, critiques, builds on another's idea

The hypothesis is that outcomes improve monotonically: I > C > A > P.

*Skeptic: That's a 2014 paper. Has it been replicated? Can you actually measure these modes from chat text alone?*

Fair point. The ICAP hierarchy is well-supported for in-person learning contexts. Whether the same hierarchy holds in AI-mediated asynchronous chat is exactly what this study is designed to test — we are not assuming the hypothesis is true, we are providing the infrastructure to test it rigorously.

As for measuring modes from text: we do not claim to measure them perfectly. We measure 8 cognitive dimensions (C1–C8) as a proxy, scored by an ensemble of 4 LLMs with inter-rater reliability metrics. The κ values will tell us whether the scoring is credible.

---

## Slide 4 — Workflow Overview

The system has three temporally separated phases:

1. **Ingestion** (offline, once): PISA PDFs → structured knowledge base
2. **Tutoring** (real-time): Socratic chatbot session, conversation logged to SQLite
3. **Evaluation** (offline, after sessions): 4 LLMs score each student turn on C1–C8

*Skeptic: Why three phases? Why not score in real-time?*

Because if the chatbot knows what dimensions a student is scoring low on, it will subtly adapt its questions to probe those dimensions — which makes the assessment circular. This is the Hawthorne Effect. NECTEC required strict separation as a condition of the research design. The chatbot literally has zero imports from the evaluation layer — that wall is enforced at the code level, not just as a policy.

---

## Slide 5 — Phase 1: Ingestion Pipeline

We process PISA PDFs using a Vision Language Model — Qwen2.5-VL-7B — rather than OCR.

*Skeptic: Why not just use a PDF text extractor? PyMuPDF can pull text directly.*

PISA documents are the worst case for text extraction: multi-column layouts, embedded figures, LaTeX equations, Thai and Chinese text with diacritics, and tables whose structure is destroyed the moment you extract raw characters. We tried EasyOCR in v1. Thai scientific text accuracy was under 70%. Equation extraction was essentially random.

The VLM reads the page as an image and outputs structured JSON directly — topic, correct concept, common misconceptions, key vocabulary, sub-questions. No OCR intermediate step means no OCR error accumulation.

*Skeptic: A 7B model doing complex structured extraction at low temperature — aren't hallucinations a real concern?*

Yes. That is why we validate every record before writing it to the knowledge base. Required fields must be present. Type mismatches are normalised. Records that fail validation are discarded with a warning. We also log everything, so a researcher can audit the extraction quality per PDF. This is not a solved problem — it is a known limitation that we mitigate, not eliminate.

---

## Slide 6 — Phase 1: Knowledge Representation

From the extracted JSONL we build two parallel structures:

- **Qdrant vector database** — BGE-M3 embeddings (dense + sparse) for semantic retrieval
- **NetworkX knowledge graph** — structural relationships: misconceptions, competency links, concept neighbours

*Skeptic: Why a knowledge graph on top of a vector database? Isn't that over-engineering?*

Vector search tells you "this passage is semantically similar to the query." It does not tell you "this concept is commonly confused with that one" or "understanding X is a prerequisite for Y." The graph encodes those structural relationships. When a student asks about photosynthesis, the tutor retrieves not just similar passages, but the full misconception neighbourhood — "plants get energy from soil," "oxygen comes from CO2 splitting." That directly improves the Socratic questioning quality.

*Skeptic: How do you know the graph is accurate? Did you validate the edges?*

The edges are derived automatically from the VLM extraction output — misconceptions come from the `common_misconceptions` field, vocabulary relations come from shared key terms. The graph inherits whatever accuracy the VLM extraction achieves. We have not done a manual edge validity audit, and that is a legitimate gap.

---

## Slide 7 — Phase 2: Socratic Tutoring

The chatbot uses Qwen2.5-7B-Instruct via Ollama. It retrieves context from both the vector database and the knowledge graph, then generates Socratic questions — not answers.

*Skeptic: How do you ensure the tutor is actually Socratic and not just answering questions?*

Two layers. First, the system prompt instructs the LLM to never give direct answers. Second, we maintain a forbidden word list — "correct," "wrong," "well done," "ถูกต้อง," "正确," and equivalents in all three languages — and strip any that appear in the response at generation time. The regex runs after every generated token batch.

*Skeptic: That sounds fragile. Can't the model say "that's a good line of thinking" without using any forbidden words?*

Partially true. The regex catches explicit judgment words but cannot catch all implicit reinforcement. This is an active research limitation. The system prompt framing is the primary guardrail; the regex is a secondary safety net. Whether subtle reinforcement language leaks through and biases behaviour is something the response time analysis could surface — if students respond faster when they get implicit praise, that would show up as noise in the duration-ICAP correlation.

---

## Slide 8 — Phase 2: Response Time as Cognitive Proxy

We record how long a student thinks before replying — the duration between when the AI finishes streaming and when the student submits their message.

| Duration | Assumed ICAP Level |
|----------|-------------------|
| < 8 seconds | Passive |
| 8–25 seconds | Active |
| > 25 seconds | Constructive / Interactive |

*Skeptic: Those thresholds look completely arbitrary. Where do they come from?*

They are informed by the research literature on think-time in educational contexts, but you are right that they are not empirically calibrated for this specific platform or population. They are a starting hypothesis, not a validated instrument. The plan is to use the scored data to calibrate them retroactively — do students who score high on C4–C8 actually cluster above 25 seconds? If not, we revise. These thresholds should not be interpreted as ground truth.

*Skeptic: Also — response time measures typing time, not thinking time. A student could write a shallow response slowly.*

Correct. Duration is a behavioural proxy, not a direct cognitive measure. The central analysis correlates duration with C4 score (causal reasoning), not uses duration as a dimension in itself. A weak correlation would tell us the proxy is unreliable. A strong one would provide supporting evidence. We are transparent about this distinction.

---

## Slide 9 — Phase 3: Offline Evaluation — The Ensemble

After sessions end, four locally-running LLMs independently score each student turn on C1–C8 (scored 0/1/2 per dimension):

| Model | Strength |
|-------|---------|
| Qwen2.5-7B-Instruct | General reasoning |
| Mistral-7B-Instruct-v0.2 | Scientific reasoning |
| Llama3.1-8B-Instruct | Instruction following |
| Typhoon2-8B-Instruct | Thai-tuned |

Their votes are aggregated using confidence-weighted plurality voting. The final score per dimension is the weighted dominant vote.

*Skeptic: Using LLMs to score LLM-generated conversations seems circular. The evaluators might share the same biases as the tutor.*

This is the most serious methodological concern in the system. The mitigations are: (1) different model families — Mistral and Llama are architecturally distinct from Qwen; (2) Typhoon2 is fine-tuned specifically for Thai, adding domain-specific orthogonality; (3) inter-rater reliability (Fleiss' κ) will reveal if they are all agreeing for the wrong reasons — systematic bias shows up as κ near zero even with high nominal agreement. If all four models share a bias, κ will be high but calibration error will be high too. We measure both.

*Skeptic: Why 4 models? Why not just use one large model — GPT-4, Claude 3 Opus?*

Three reasons: cost at research scale (hundreds of sessions, 12 turns each, 8 dimensions = tens of thousands of API calls); reproducibility (cloud model versions change silently); and the ensemble's theoretical advantage — majority voting across 4 diverse 7–8B models outperforms a single 30B model on ordinal classification because the models' failure modes are decorrelated. This claim has empirical support in the NLP literature. It is not just a budget justification.

---

## Slide 10 — The Kappa-Adaptive Weighting Loop

Once enough data accumulates, we compute each model's historical agreement with the majority vote and adjust its weight for future scoring. Models that consistently agree with the consensus get more voting power.

*Skeptic: Isn't this circular? You're weighting models by how much they agree with each other, not by how much they agree with ground truth.*

Yes, and this is a known limitation of any unsupervised ensemble. Weights are initialised at 1.0 (equal) and adjust slowly. The adaptive weighting is most useful for detecting a model that has started drifting — if Ollama updates a model silently and it starts returning different scores, its agreement with the others will drop and its weight will fall. For ground-truth validation, we have a separate optional phase: DSPy prompt optimisation with ~50 manually annotated gold examples. Without those gold labels, adaptive weighting is a consistency measure, not an accuracy measure. We are explicit about this distinction.

---

## Slide 11 — Inter-Rater Reliability

We report Fleiss' κ and Krippendorff's α per dimension. Our target is κ ≥ 0.70 on C4–C8.

*Skeptic: Why is 0.70 the target? That's a fairly moderate bar for a claim about measuring cognitive depth.*

0.70 corresponds to "substantial agreement" in the Landis & Koch scale — it is the conventional threshold for acceptability in educational measurement. The honest answer is that C4–C8 (causal reasoning, evidence evaluation, metacognition) are genuinely hard to operationalise, and expecting near-perfect agreement on nuanced constructs is unrealistic. If we consistently achieve κ < 0.40, that would be evidence the dimensions are not reliably measurable from chat text alone — a meaningful null result, not a failure.

---

## Slide 12 — Hardware Reality Check

*Skeptic: This system requires a 24 GB GPU, 4 Ollama models, Qdrant, a FastAPI server, and a Next.js frontend. What research institution can run this?*

The 24 GB GPU requirement is for the ingestion phase — Qwen2.5-VL-7B in bfloat16. That runs once per knowledge base build. Ongoing session serving only requires the Ollama models (can run on 16 GB with quantisation) and the API/frontend stack (CPU-only). The evaluation pipeline runs overnight and tolerates slow inference.

The system runs fully air-gapped — no cloud API keys required. For institutions without GPU resources, the ingestion step can be run on a rented cloud machine once, and the knowledge base assets shipped to the local deployment. It is not plug-and-play, but it is more accessible than it looks at first pass.

---

## Slide 13 — What We Do Not Know Yet

Let me be direct about the open questions this study cannot yet answer:

1. **Does the ICAP hierarchy hold in AI-chat contexts?** We are testing this, not assuming it.
2. **Are the C1–C8 rubric dimensions the right decomposition?** They are grounded in Chi & Wylie but adapted for chat. The rubric may need revision after seeing real data.
3. **Does Socratic tutoring increase higher-order engagement compared to direct instruction?** We have no control condition in this design.
4. **Are the duration thresholds valid for Thai student populations?** Calibration requires data we do not yet have.
5. **Do the LLM evaluators share systematic biases that κ cannot detect?** This requires gold-standard manual annotation to rule out.

*Skeptic: So what exactly are you claiming to contribute?*

An open-source, reproducible infrastructure for cognitive engagement measurement at scale — with inter-rater reliability and calibration metrics built in — that makes the above questions empirically testable. The contribution is the measurement instrument, not the answer. The answers come from using it.

---

## Slide 14 — Closing

The PISA Cognitive Tutor is a research platform, not a product. It makes one core bet: that the gap between "a student who copies a definition" and "a student who constructs an explanation" is detectable in their language, and that detecting it reliably at scale is worth doing.

Whether that bet pays off will show in the κ values, the duration correlations, and whether C4–C8 scores predict the transfer-learning outcomes we measure downstream.

If the κ values come back below 0.40 — we will say so, publish the null result, and redesign the rubric. That is what the reliability infrastructure is for.

Thank you.

---

## Appendix: Anticipated Hard Questions

**Q: Why SQLite instead of PostgreSQL?**
Single-node research tool. SQLite with WAL mode handles concurrent reads from the API and writes from the session manager without issue at research-session scale (tens of concurrent students, not thousands). Zero ops overhead.

**Q: Why not fine-tune one model instead of using an ensemble?**
Fine-tuning requires gold-labelled training data we do not have at the start of the study. The ensemble is a zero-shot approach. Once we have ~500 manually annotated examples, fine-tuning is a valid next step.

**Q: The forbidden word list looks brittle — what about Thai synonyms the regex misses?**
The `icap_rubric.yaml` config exposes the forbidden word list. Expanding it is a maintenance task, not an architectural one. We welcome contributions from Thai-language educators who can identify praise/judgment synonyms we missed.

**Q: Why track only 12 turns per session?**
Research-validated cutoff for reliable cognitive scoring — too few turns and the profile is noisy; too many and fatigue effects distort the trajectory analysis. 12 is a common boundary in educational dialogue research for this reason. It is configurable in `model_configs.yaml`.

**Q: Gemini API appears in the codebase — is student data sent to the cloud?**
Gemini is a disabled-by-default fallback for knowledge base synthesis only. No student conversation data is ever sent to any cloud API. The evaluator ensemble, the chatbot, and all inference run on local Ollama. This is verifiable by reading `tutor_engine.py` — there are no external HTTP calls except to `localhost:11434`.

---

*Script version: 2026-05-14 — for NECTEC research presentation*
