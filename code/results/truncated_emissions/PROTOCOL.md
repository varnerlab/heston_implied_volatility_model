# Truncated Student-t price simulations

Decision recorded September 14, 2026, before rerunning the affected experiments.
The author authorized explicitly truncated Student-t emissions for stock-price
simulations. The saved neural IV surfaces and the 2014–2024 JumpHMM fit are retained.

For each fitted state k, draw G = mu_k + sigma_k Z with Z distributed as
Student-t(nu_k) conditional on -b <= Z <= b. The primary cutoff is b = 10;
the wider sensitivity case is b = 20. These are explicit stress assumptions,
not fitted quantiles or cutoffs selected to improve 2025/2026 forecast scores.
They use the training-fitted state locations and scales. A scale unit is not
a standard deviation: for the original t(5), SD = scale * sqrt(5/3).
The bounds are state-specific growth-rate bounds, not a universal percentage
cap on daily stock returns. Report the resulting daily bounds separately.

Generate the original state/jump path and emission proposals with the pinned
JumpHMM version. Independently replace only out-of-support emissions with draws
from the conditional Student-t distribution, before drift normalization and
exponentiation. Keep all stock paths. This gives the conditional emission law
without point masses at the boundaries. It retains the original state paths,
jump flags, and accepted proposals for a fixed seed. Never clip stock prices,
option marks, or P&L based on forecast results.

Recompute each experiment's pilot or ensemble normalization under its specified
cutoff. Preserve the existing chronological cutoffs, dates, contracts, seeds,
path counts, and fitted surface inputs. Update simulation cache identity and
record the emission specification and source hashes in result manifests.
The Student-t law used for historical state inference is also conditioned on
these bounds; unsupported observations must be reported, not silently clipped.

Regenerate the fitted GS/LLY illustrations, paired IV ablation, both chronological
option cohorts, stock-initialization diagnosis, and frozen stock benchmark
comparison. Run the wider-cutoff sensitivity on stock and option outcomes with
the same seeds and contracts. Compare means, lower-tail call P&L, quantiles,
and forecast errors, and report sensitivity rather than assuming it is small.
Historical source snapshots remain associated with their historical results;
new results must retain the exact source that produced them.

At any fixed horizon the bounded emissions, finitely many states, and finite
normalization constants imply bounded log returns and finite stock-price moments.
This removes the divergent-moment problem. It does not validate the cutoff,
establish a forecasting improvement, or remove other model limitations.
