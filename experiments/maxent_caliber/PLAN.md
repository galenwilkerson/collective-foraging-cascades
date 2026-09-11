# Experiment: the two maximum entropies — does the max-caliber search exponent track the max-entropy field?

**Purpose (for the paper, NOT the live demo).** Make the "two maximum entropies" coincidence quantitative:
maximum *entropy* fixes *where resources sit* (a sparse, scale-free clustered field); maximum *caliber* fixes
*how to move* (a heavy-tailed Lévy search). The elegant-coincidence claim is that these are one least-committal
principle seen twice. The sharp, testable version: **the Lévy search exponent that is optimal for foraging a
scale-free resource field tracks the field's own scaling exponent** (the Lévy-flight foraging hypothesis,
Viswanathan et al. 1999). A small live demo (16 targets, bounded torus) cannot show this — no dynamic range —
so it is done here at scale.

## Hypotheses
- **H1 (heavy tails win).** In a sparse, clustered (scale-free) field, the encounter-rate–optimal search is
  heavy-tailed (1 < α* < 3), beating the Brownian limit (α→3, Gaussian-like). 
- **H2 (matching / tracking).** The optimal search exponent α*(β) increases monotonically with — ideally
  matches — the field's scaling exponent β. A match α*≈β is the strongest form of the coincidence; a clean
  monotone tracking is the weaker, still-publishable form.
- **Null.** α* is independent of β (no coupling between environment and optimal search).

## Environment — the maximum-entropy (scale-free) resource field
- 2-D torus, side `L` (default 1000), `M` point targets (default 600).
- **Lévy-dust construction** (standard, controllable fractal): place targets as the visited sites of a Lévy
  walk with step-length exponent `β` (P(ℓ)∝ℓ^{−β}, 1<β<3). The resulting point set is statistically
  self-similar with fractal (box-counting) dimension `D_f ≈ β − 1`. Sweeping β sweeps clustering from strong
  (β→1, near-ballistic dust, very clumped) to weak (β→3, near-Poisson/uniform).
- **Measured field exponent** (do not trust the input β blindly): recover the field's scaling from the data by
  (a) box-counting fractal dimension D_f, and (b) the CCDF slope of pairwise/nearest-neighbour distances. Report
  the measured exponent, use it on the x-axis.

## Searcher — the maximum-caliber (Lévy) search
- Single searcher on the same torus; step lengths ℓ ~ power law with exponent `α` (Mantegna/inverse-transform),
  minimum step `ℓ0` (= detection radius `rv`), direction uniform. **Non-destructive/revisitable** targets
  (classic case; also run destructive as a robustness check).
- Detection: a target is encountered if the search *segment* passes within `rv` of it (point-to-segment
  distance ≤ rv), not just the endpoints — segment test matters for long hops. Use a cKDTree of targets and
  query along each hop.
- Budget: fixed total path length `Λ` (default 2e5 · rv) so all α are compared at equal travel cost (the
  honest efficiency denominator).

## Efficiency metric
- η(α) = (number of target encounters) / (total path length). Non-destructive → count every encounter;
  destructive → count distinct targets, remove on encounter. Primary metric = η; also report distinct-target
  coverage.

## Procedure
1. β grid: {1.3, 1.6, 1.9, 2.2, 2.5, 2.8}. α grid: 11 values in [1.1, 3.0].
2. For each β: `Rf` field realizations (default 6). Measure D_f and distance-CCDF slope per field.
3. For each (field, α): `Ks` searcher replicas (default 8), random starts. Compute mean η.
4. α*(β) = argmax_α η̄(α,β), with a parabolic fit around the peak for a continuous estimate; bootstrap CI over
   field×searcher replicas.
5. **Figures:**
   - **(a) η(α) curves**, one per β — peaks should shift with β (visual of H1/H2).
   - **(b) α*(β)** vs measured field exponent, with the identity line α*=β (test of H2) and bootstrap CIs.
   - (opt.) CCDF overlay of search-step lengths vs field inter-target gaps at the matched α*, on one field.

## Controls / robustness
- Poisson (uniform) field as a β→∞ control: expect flat-ish η(α) or optimum drifting toward the ballistic end.
- Vary rv and M (density) — the classic result is density-dependent, so report η(α) peak vs density too.
- Destructive vs non-destructive.
- Seeds logged; multiple realizations; bootstrap CIs; no single-run claims.

## Collective extension (second figure, optional for this pass)
Repeat (a)/(b) with the full swarm + recruitment (the paper's model) instead of a lone searcher, to ask
whether the *collective* effective search inherits the same optimum-tracking. Deferred unless the single-searcher
result is clean.

## Deliverables
- `run_maxent_caliber.py` — self-contained (numpy + scipy.cKDTree); writes `fig_eta_vs_alpha.pdf`,
  `fig_alphastar_vs_beta.pdf`, and a `results.npz` + `results.md` summary with the fitted relationship and CIs.
- Runtime target: minutes on CPU. Reproducible (fixed seed grid).
