# 🧠 AI Resume Information Extraction with Named Entity Recognition

**Fine-tuning DistilBERT for token classification to extract structured information from raw resume text.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-latest-yellow.svg)](https://huggingface.co/docs/transformers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Overview

Recruiters and Applicant Tracking Systems receive thousands of resumes per posting, each in a different
layout and writing style. This project builds an end-to-end NLP pipeline that converts unstructured resume
text into structured, queryable records — the core technology behind ATS auto-fill, candidate search, and
talent analytics.

The pipeline fine-tunes **DistilBERT** on **5,960 annotated resumes** (Kaggle Resume NER Training Dataset)
across **14 entity types** (PERSON, DESIGNATION, COMPANY, SKILL, EDUCATION, EMAIL, LOCATION, EXPERIENCE, …)
and wraps the trained model in a reusable, confidence-scored inference API with JSON/CSV export.

## Results

Evaluated with strict **entity-level** metrics (`seqeval`): a prediction counts only if the entire span
*and* type match the gold annotation.

| Metric (test set) | Score |
|---|---|
| **F1** | **0.737** |
| Precision | 0.712 |
| Recall | 0.764 |
| Token accuracy | 0.913 *(inflated by the dominant `O` class — reported for completeness only)* |

Validation F1 (0.735) and test F1 (0.737) are nearly identical, confirming no train/test leakage —
duplicates are removed *before* splitting.

Per-entity highlights: **EMAIL 0.88** (formulaic, easy to learn), **SKILL 0.75** (94% of all entity
mentions), while vague catch-all classes (OTHER, EXPERTISE, ACTION) score far lower — see
[Limitations](#limitations--honest-caveats).

### Sample extraction

```
Input : "Michael Chen, Software Engineer II at Shopify, Toronto (2021 - Present)
         B.Sc. Computer Science, University of Waterloo, 2019 …"

Output:
  PERSON       Michael Chen                                (conf 0.84)
  DESIGNATION  Software Engineer                           (conf 0.96)
  LOCATION     Toronto, Ontario                            (conf 0.95)
  EXPERIENCE   (2021 - Present)                            (conf 0.98)
  EDUCATION    Bachelor of Science in Computer Science     (conf 0.98)
  EMAIL        michael.chen@outlook.com                    (conf 0.93)
```

## Pipeline

```
ZIP upload ─► format auto-detection ─► EDA ─► cleaning & de-dup ─► entity-safe chunking
    ─► BIO encoding ─► WordPiece tokenization + subword label alignment
    ─► DistilBERT fine-tuning (warmup, fp16, early stopping, best-F1 checkpointing)
    ─► entity-level evaluation + confusion matrix ─► error analysis
    ─► inference API with confidence scores ─► JSON / CSV export ─► saved model artifact
```

Notable engineering details:

* **Format-agnostic ingestion** — a universal loader auto-detects spaCy-array, Dataturks-JSONL, doccano,
  CoNLL, and token/tag-CSV formats, so any mirror of the dataset works without path or code edits.
* **Data quality before modeling** — invalid BIO tags repaired, dangling `I-` tags promoted, unpaired
  Unicode surrogates sanitized (offset-preserving), exact duplicates removed *before* splitting, and long
  resumes chunked into ≤150-word windows with boundaries that never cut through an entity.
* **Honest evaluation** — best checkpoint selected by validation F1; the test set is touched exactly once.
* **Self-describing artifact** — label mappings ship inside the model config; reload is one line.

## Quick Start

1. Download the Resume NER Training Dataset from Kaggle as a ZIP.
2. Open `Resume_NER_DistilBERT_Pipeline.ipynb` in Google Colab.
3. `Runtime ▸ Change runtime type ▸ GPU`, then `Runtime ▸ Run all`.
4. Upload the ZIP when prompted in Section 3. That is the only manual step.

Training takes ~15 minutes on a free Colab T4. After training, extract entities from any resume:

```python
info = extract_resume_info(raw_resume_text)          # {"PERSON": [{"text": ..., "confidence": ...}], ...}
export_predictions_json({"resume_001": raw_resume_text})
export_predictions_csv({"resume_001": raw_resume_text})
```

## Tech Stack

Python · PyTorch · Hugging Face Transformers / Datasets / Evaluate · seqeval · pandas · NumPy ·
matplotlib · scikit-learn

## Limitations — honest caveats

* **Severe class imbalance.** SKILL accounts for ~94% of entity mentions, so the overall F1 largely
  reflects SKILL performance. Rare classes (CERTIFICATION: 1 test example; ACTION: 8) cannot be learned
  or reliably measured from this data.
* **Annotation noise sets the ceiling.** Error analysis shows gold labels that tag entire job-description
  paragraphs as a single SKILL entity, and unlabeled spans ("network security", "VPN") where the model's
  prediction is arguably more correct than the annotation. The dominant confusion pair (O ↔ SKILL,
  ~14K tokens) is mostly annotator inconsistency, not model failure — better labels, not bigger models,
  are the highest-ROI improvement here.
* **Output artifacts.** Whitespace tokenization occasionally leaves trailing punctuation on extractions
  ("Python,"), and the model sometimes tags section headers (EDUCATION, SKILLS:) as skills — it learned
  that tokens *near* skill lists are skills. Both are addressable in postprocessing.
* **Plain text only.** Real resumes live in PDF/DOCX with informative layout this model never sees.
* **Uncalibrated confidences.** Softmax scores rank predictions usefully but are not true probabilities.

## Future Work

Label-space cleanup (merge EXPERTISE→SKILL, drop catch-all classes) and retraining · CRF decoding head to
reduce boundary errors · RoBERTa / DeBERTa-v3 encoders · LayoutLMv3 + OCR for PDF resumes ·
confidence calibration · Streamlit demo with highlighted entities · ATS integration.

## Ethics Note

Resume parsing sits inside hiring pipelines — a regulated, high-stakes ML application (NYC Local Law 144,
EU AI Act). Extracted fields such as names, locations, and graduation years are proxies for protected
attributes; any downstream ranking system must be audited for disparate impact, keep humans in the loop,
and handle resumes as personal data under GDPR/CCPA-style regimes.

## License

MIT — dataset © its original Kaggle authors; see the dataset page for its terms.
