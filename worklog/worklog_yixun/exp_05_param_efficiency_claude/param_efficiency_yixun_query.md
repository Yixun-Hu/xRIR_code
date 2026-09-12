# Queries — param_efficiency (exp_05)

## Query 1 (2026-09-12, verbatim, translated context)
> Could you please tell me how to do this experiment? "Parameter efficiency refers to how much prediction performance is obtained with how many parameters … (1) same parameter budget, compare performance; (2) different budgets, draw the performance–parameter curve (Small/Medium/Large for baseline and CERPA, x = parameters, y = unseen-room acoustic error); (3) fixed performance target, compare the parameters needed … For CERPA, first add parameter statistics (encoder, full system, trainable) to the existing table; the parameter-matched main table should exist; the three-tier capacity curve is for when parameter efficiency is to be emphasised." on xRIR on AcousticRooms dataset

## Query 2 (2026-09-12, verbatim)
> I think you should run the experiment 2 you said before first, please write the experiment plan and I will ask another claude code agent to run it.

**Summary.** Run the capacity curve (experiment 2 of the note) for xRIR on AcousticRooms: three encoder tiers × two backbones (SimpleViT baseline vs CylindricalViT = CERPA inside the same acoustic system), same recipe, same step budget; plot unseen-room error against encoder parameters; read the fixed-target comparison (experiment 3) off the curve.

**User's hypothesis.** CylindricalViT reaches lower unseen-room acoustic error than SimpleViT at every parameter budget (parameter efficiency), and a smaller cylindrical encoder can match a larger baseline encoder.

**Why it needs to run.** The matched-budget comparison exists (exp_01/exp_03: the two encoders differ by 0.24 % in parameters; cylindrical −4.4 % EDT at K = 8, no C50 difference, worse at K = 1); a parameter-efficiency claim needs the curve.

**Handoff.** This plan is written by the Planner of exp_04's session for another Claude Code agent to execute under `worklog/experiment_SOP.md`.
