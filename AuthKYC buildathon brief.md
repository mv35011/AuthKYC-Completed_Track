# AuthKYC → Razorpay AI Buildathon Submission (AI Risk Manager Track)

## Context for the agent

This repo (`AuthKYC--END-to-END-system-for-bank-kyc-against-AI-attacks`) is a working
4-stage Presentation Attack Detection (PAD) pipeline for video-based bank KYC, built as
a college minor project. It already achieves 95.2% accuracy / 0% false-positive rate on
a 21-video mixed evaluation set. The core detection logic is good and should **not** be
rewritten from scratch. Your job is to:

1. Fork this into a new branch/repo (`authkyc-buildathon`) — do not touch the original
   academic repo's history or its ongoing benchmark work.
2. Re-skin the framing from "PAD research" to "fraud-loss prevention for KYC onboarding."
3. Close specific credibility gaps (small eval set, latency, missing cost framing).
4. Add a small number of high-impact "wow" features for a live judge demo — scoped
   tightly so they don't threaten the core pipeline's stability.

Target event: Razorpay AI Buildathon, track **AI Risk Manager** — *"Build a working
detector, verifier or auto-responder for one class of loss, with measured precision and
recall on a held-out test set... Honest metrics including false-positive cost. Strictly
defense-only: anything offense-capable is disqualified."*

Deliverables required by the event: a public GitHub repo, a 5-minute pitch video, and an
architecture writeup. Deadline: applications close 5 September.

Hardware available: college server with NVIDIA A2000, AMD Threadripper, 128GB RAM, 5TB
storage. Use it — the current eval set (N=21) and CPU-bound MTCNN latency are both
compute-limited, and this hardware removes that excuse.

**Non-negotiable constraint:** this system must remain strictly defense-only. Do not add
anything that could double as an attack-generation or spoofing tool, even for internal
testing — see the "Attack Museum" feature below for how to do adversarial demos safely
(pre-recorded/curated samples, not a live generator).

---

## Workstream 1 — Close the credibility gap (do this first)

- [ ] Pull a subset of OULU-NPU and/or SiW-Mv2 onto the server and fold it into a new,
      larger held-out evaluation set (target: hundreds of clips across attack types, not
      21). Report precision, recall, and FPR on this set as the headline metric. Keep the
      original 21-video waterfall ablation as supporting "which stage caught what" detail,
      not the headline anymore.
- [ ] Re-run FTCA fine-tuning with more data/epochs now that GPU time isn't the
      bottleneck. Current best val accuracy is 80.85% — try to move this meaningfully.
      Log before/after clearly (this is also useful for the academic submission).
- [ ] Swap MTCNN for a faster face detector (RetinaFace or YOLOv8-face) to cut the
      5–15s per-video latency down to something safe for a live demo (target: under 3s on
      the A2000). Keep MTCNN as a fallback path if needed, but the demo path should use
      the fast one.
- [ ] Investigate TensorRT/INT8 quantization for the FTCA stage if time allows — this
      was already flagged as future work in the report; now it's actually feasible.

## Workstream 2 — False-positive cost framing

The track's evaluation bar explicitly asks for "honest metrics including false-positive
cost." Build this as a first-class artifact, not a paragraph buried in the report:

- [ ] Write a short cost model: cost of a false reject (real customer blocked mid-KYC →
      support escalation, drop-off, lost merchant) vs. cost of a false accept (fraud
      onboarded → chargeback/regulatory exposure).
- [ ] Expose this as an actual UI control (see Workstream 4, feature #1) — don't just
      state it in the README, let a judge move a slider and see the system's decision
      boundary shift in response to different cost assumptions.
- [ ] Document explicitly where the existing Dynamic Fallback mechanism (S1 override
      when S3+S4 pass) already embodies this tradeoff — it exists to avoid over-rejecting
      legitimate mobile users, which is exactly false-positive-cost thinking. Call this
      out in the pitch; it's a genuine design decision, not an afterthought.

## Workstream 3 — Reframe, don't rewrite

- [ ] New top-level README framed around: *"A pre-onboarding fraud gate that stops
      deepfake / synthetic-identity account-takeover before a fraudulent customer or
      merchant clears KYC — with a stage-by-stage, auditable decision trail."*
- [ ] Keep all technical section headers (PRNU, Moiré, rPPG, FTCA) but reframe their
      intros from "physical principle exploited by attackers" to "fraud signal we check
      before a customer is approved."
- [ ] Condense `project_report.tex` into a short product-voiced architecture doc for the
      buildathon submission — separate file, don't overwrite the academic report.
- [ ] The existing `/api/v1/audit_stream` endpoint's structured JSON output (per-stage
      pass/fail + scores) *is* the audit trail the event's bar language asks for across
      every track. Make sure this is front-and-center in the demo, not buried.

## Workstream 4 — Demo wrapper + judge-facing features

Build a lightweight web UI (Streamlit or a small React/FastAPI app is fine — reuse
existing FastAPI backend) in front of the existing pipeline. Core flow: upload a clip →
watch the four-stage waterfall evaluate live → see final verdict + audit JSON.

On top of that core flow, implement these in priority order — stop if you run out of
time, don't sacrifice pipeline stability for these:

### 1. Fraud-risk score + cost-slider dashboard (highest priority)
Replace (or add alongside) the binary pass/fail with a continuous 0–100 fraud-risk score
derived from the four stage outputs. Add a slider for "cost of false reject" vs. "cost
of false accept" and show the decision threshold moving live as the judge drags it. This
directly demonstrates the false-positive-cost framing from Workstream 2 in an
interactive, memorable way — this is the single feature most likely to make a judge
sit up, because it answers the track's stated bar in a way most teams will only put in
a paragraph.

### 2. Live explainability overlay
While a clip processes, show:
- the PRNU noise-residual heatmap,
- the Moiré 2D FFT spectrum with the peak-ratio annotated,
- a live-updating rPPG waveform plot with BPM/SNR readout.
This turns "trust the model" into "watch the physics happen" — much stronger for a
5-minute pitch than a final score alone.

### 3. Attack Museum (curated, not generative)
A small pre-recorded gallery (5–8 clips) of known consumer spoofing methods — OBS
virtual camera injection, phone screen replay, a DeepFaceLive-style face swap, a
diffusion-generated talking head — each pre-labeled with which stage caught it and why.
Let a judge click through these instead of needing to bring their own attack video live.
**Important:** these clips must be pre-recorded/curated ahead of time, not generated live
by the demo app — the system stays purely defensive, nothing in the shipped repo should
generate spoofing content on demand.

### 4. Merchant-style audit ledger view
A simple dashboard listing recent verification sessions (mocked/synthetic data is fine)
with pass/fail, risk score, and stage-by-stage audit trail per row — framed as what a
Razorpay-style merchant-onboarding team would actually look at. Reinforces the "audit
trail" language from the event's bar and makes the fraud-prevention framing concrete.

### 5. (Stretch, only if time remains) Live latency/throughput telemetry
A small panel showing per-stage processing time on the A2000 in real time, and
before/after numbers versus the original CPU-bound MTCNN pipeline. Good proof of the
"we productionized this" story but lowest priority — cut first if time is tight.

---

## Repo hygiene

- [ ] New branch/repo, clearly named for the buildathon, separate from the academic repo.
- [ ] README rewritten per Workstream 3; keep a link back to the original academic
      report for anyone who wants the full research writeup.
- [ ] `.env`, model weights, and dataset paths should not be committed — check
      `.gitignore` before pushing (the current repo appears to have a `.env` file
      tracked; fix this).
- [ ] Add a short `DEMO.md` with exact steps to reproduce the live demo flow, in case
      the panel wants to run it themselves.

## Out of scope / do not build

- Any live/generative deepfake or spoofing capability, even for testing — use curated
  pre-recorded samples only (see Attack Museum above).
- Real Razorpay API integration — the merchant-onboarding framing is narrative only,
  no real payment rails needed.
- A full mobile app or production auth system — this is a demo-grade wrapper around an
  existing research pipeline, not a new product build.