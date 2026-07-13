# Universal approximation for FC logic networks

This note formulates the sense in which a **fully-connected logic network**
({class}`~silogic.LogicNet`) with a **`GroupSum` head** is a *universal
approximator*, and states the result for two representative node arities used in
the library — **arity 2** (`node="gate16"`) and **arity 6** (an
`arity=6` LUT node: `"multilinear"` / `"hybrid"` / `"walsh"`).

The statements below are about the **hard (deployed) circuit**
(`forward_hard`) — what the discretized network can *represent*. Trainability
(whether SGD finds these weights) and the soft→hard discretization gap are
separate questions, discussed at the end.

## Setup

Inputs are binary, $x \in \{0,1\}^n$ (this is what logic nets consume — the
thermometer-binarized features). A hard {class}`~silogic.LogicNet` computes, at
each layer, a width-$W$ vector of Boolean values, where every node is a Boolean
function of the `arity` wires its connectome selects from the previous layer.
The {class}`~silogic.GroupSum` head splits the final width-$W$ layer into $C$
equal blocks and outputs, per class $c$, the block popcount

$$s_c(x) \;=\; \sum_{j \in \text{block}_c} y_j(x), \qquad y_j(x)\in\{0,1\},$$

and the prediction is $\hat g(x) = \arg\max_c s_c(x)$ (`tau` is an
argmax-invariant positive scale). So the model is exactly a

> **layered, bounded-fan-in Boolean circuit → per-class popcount → argmax.**

Because the domain $\{0,1\}^n$ is **finite**, "universal approximation" is not a
density statement in a function space — it is *exact representation*: can the
architecture compute **every** classifier $g:\{0,1\}^n \to \{1,\dots,C\}$?

## The theorem

**Theorem (universal representation on the Boolean cube).**
Let the node family be *Boolean-complete at its arity* — i.e. a single node can
realize a functionally complete gate (e.g. NAND). Then for **every** target
$g:\{0,1\}^n \to \{1,\dots,C\}$ there is a hard `LogicNet` with a `GroupSum`
head that computes $\hat g = g$ *exactly*.

The completeness hypothesis holds for both arities we care about:

| node | arity | hard function class per node | complete? |
|---|---|---|---|
| `gate16` | 2 | **all 16** two-input Boolean functions | ✓ (contains NAND, gate 14) |
| `multilinear` / `hybrid` | 6 | **all $2^{2^6}$** six-input functions (a full `LUT_6`) | ✓ (a fortiori) |
| `walsh` (arity 6) | 6 | $\operatorname{sign}$ of a full $2^6$-coeff multilinear form = **all** six-input functions | ✓ |
| `linear` | any | only linearly-separable (threshold) functions | ✗ — *not* complete |
| `polynomial` (deg $d$) | any | only degree-$d$ threshold functions | ✗ unless $d=$ arity |

So **arity 2 (`gate16`) and arity 6 (`multilinear`/`hybrid`/`walsh`) both give a
universal architecture**; `linear` and low-degree `polynomial` nodes do **not**
(they cannot even represent XOR of two inputs, so they are genuinely weaker).

### Proof

1. **The body computes the class indicators.** For each class $c$ define the
   Boolean indicator $b_c(x) = \mathbb{1}[\,g(x)=c\,]$. A Boolean-complete gate
   basis is functionally complete, so a layered fan-in-`arity` circuit can
   compute each $b_c$ (e.g. build its DNF: OR of the min-terms where $g(x)=c$;
   AND/OR/NOT are all in `gate16`, and any single `LUT_6` computes a 2-input
   NAND by ignoring four inputs). A strictly-layered net carries intermediate
   values forward with pass-through nodes (`gate16`'s `A` gate, or an identity
   `LUT`), which is exactly what `residual_init` / `wire_residual` provide.

2. **The head routes indicators to argmax.** Make the last layer width
   $W = C\cdot m$ ($m\ge 1$ nodes per block) and set every node in block $c$ to
   output $b_c(x)$. Then $s_c(x) = m\, b_c(x)$. Because $g$ is a function,
   exactly one indicator is $1$, so $s_{g(x)} = m > 0$ and $s_{c}=0$ for
   $c\ne g(x)$ — no ties, and $\arg\max_c s_c = g(x)$. Even $m=1$ suffices. ∎

The construction needs a connectome that can wire a node to the specific
previous-layer nodes its DNF requires. `connectome="dense"` can (its hard select
is an argmax over *all* previous nodes); `topk`/`fixed` draw **random** candidate
pools, so for them the theorem holds *with high probability* over the wiring
once width and `k` are large enough that every required wire lands in some node's
candidate set.

## Arity 2 vs. arity 6: what changes

Both arities are universal, so the difference is **cost**, not capability. The
worst-case bounds are the classical circuit-complexity ones.

- **Depth.** An `arity`-$r$ LUT absorbs $\log_2 r$ levels of 2-input gates, so a
  function needing 2-input depth $\approx n$ needs LUT-$r$ depth
  $\approx n/\log_2 r$. Arity 6 is therefore about $\log_2 6 \approx 2.585\times$
  **shallower** than arity 2 — which is why the FPGA export collapses arity-2 gate
  fabric to very few LUT levels, and why deep 2-input trees lean on
  `residual_init`.

- **Size (worst case, Shannon–Lupanov).** Almost every $g:\{0,1\}^n\to\{0,1\}$
  needs $\Theta(2^n/n)$ gates *regardless of arity*; for $C$ classes,
  $\Theta(C\cdot 2^n/n)$ nodes. Higher arity buys only a **constant factor**
  here — it does not change the exponential worst case. A counting argument makes
  the trade-off explicit: a layered net with $N = W\!\cdot\!D$ nodes of arity $r$
  encodes at most $\sim N\,(2^r + r\log_2 W)$ bits (truth table + wiring per
  node), so representing all targets forces

  $$N\,\bigl(2^{r} + r\log_2 W\bigr)\ \gtrsim\ C\cdot 2^{n}.$$

  Arity 6 contributes $2^6 = 64$ truth-table bits per node vs. arity 2's
  $2^2 = 4$, so it reaches the same capacity with a constant factor fewer nodes —
  at the price of $2^r$ learnable entries **per node** (64 vs. 4), i.e. the
  soft→hard gap and the parameter count per node grow with arity (hence the
  arity warning in the [guide](guide.md)).

**Summary.** Universality is a property of the *gate basis*, and it is already
achieved at arity 2 (`gate16` ⊇ NAND). Going to arity 6 does **not** enlarge the
representable function class — it trades a larger per-node LUT for a
constant-factor-smaller, $\approx 2.6\times$-shallower circuit.

## Scope and caveats

- **Representation, not learning.** The theorem is about what `forward_hard` *can*
  compute. It says nothing about whether gradient descent on the soft relaxation
  *finds* such a circuit, nor about generalization.
- **Exact, because the domain is finite.** On $\{0,1\}^n$ there is nothing to
  approximate — a universal architecture represents every classifier exactly.
  (The "approximation" framing returns only if inputs are treated as a
  sub-sampled/continuous set, e.g. more thermometer bits approximating a
  real-valued feature.)
- **`GroupSum` is enough.** No learned decoder is needed for universality; the
  parameter-free popcount head already routes indicator bits to a correct argmax.
  The learned heads ({class}`~silogic.heads.LearnedDecoder`) can only match this
  class, not exceed it (their inputs are the same Boolean features).
- **`linear` / low-degree `polynomial` nodes are not universal** on their own —
  they realize only threshold / bounded-degree functions and cannot represent
  XOR. Universality needs a full-LUT (`multilinear`/`hybrid`/`walsh`) or the
  16-gate (`gate16`) node.
