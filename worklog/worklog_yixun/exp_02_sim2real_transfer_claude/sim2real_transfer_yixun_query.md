# Query log — sim2real_transfer (user: yixun)

> Retrospective record for an experiment planned and launched on 2026-09-05 before the SOP was adopted. The plan *was* surfaced for and received explicit user approval in chat; no independent reviewer loop ran.

## Q1 — 2026-09-05T15:20-04:00
**Verbatim:** "After the cylindricalvit and simplevit comparison, I want to see the sim2real transfer results, before running the sim2real experiments, I want you to first simply give me your running configs and plans for me to approve."
**Summary:** Extend the backbone comparison to the paper's sim-to-real setting (Hearing Anything Anywhere rooms, paper §5.5 / Table 2), but present configs and plan for approval first.
**Assumption / hypothesis:** The CylindricalViT advantage measured in simulation carries over to real rooms after the paper's two-stage few-shot fine-tuning; the released checkpoint should reproduce Table 2.
**Why:** Table 2 is the paper's real-world claim; a backbone change is only useful if it survives fine-tuning on 12 real RIRs per room.

## Q2 — 2026-09-05T16:00-04:00
**Verbatim:** "Approved, run the sim2real experiments after training finishes"
**Summary:** Plan approved as presented (see `plan_sim2real_transfer.md`); constraint: start only after the exp_01 trainings finish.
