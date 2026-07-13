# Universal approximation for FC logic networks

This note formulates the sense in which a **fully-connected logic network**
({class}`~silogic.LogicNet`) with a linear-style head — the parameter-free
{class}`~silogic.GroupSum`, or a learned {class}`~silogic.heads.LearnedDecoder`
— is a *universal approximator*, and works out the depth, width, and arity it
costs. Results are stated for the two representative node arities in the library:
**arity 2** (`node="gate16"`) and **arity 6** (an `arity=6` LUT node:
`"multilinear"` / `"hybrid"` / `"walsh"`).

Everything below is about the **hard (deployed) circuit** (`forward_hard`) — what
the discretized network can *represent*. Trainability (whether SGD finds these
weights) and the soft→hard gap are separate questions, noted at the end.

## Setup

Inputs are binary, $x \in \{0,1\}^n$ (what logic nets consume — the
thermometer-binarized features). A hard {class}`~silogic.LogicNet` is a stack of
$D$ layers of width $W$; each node is a Boolean function of the `arity` $=r$
wires its connectome selects from the previous layer. The head maps the final
width-$W$ feature vector $\phi(x)\in\{0,1\}^W$ to class scores:

- **`GroupSum`** splits $\phi$ into $C$ equal blocks and outputs the block
  popcounts $s_c(x)=\sum_{j\in\text{block}_c}\phi_j(x)$;
- a **learned linear decoder** outputs $s_c(x)=\langle w_c,\phi(x)\rangle+b_c$.

The prediction is $\hat g(x)=\arg\max_c s_c(x)$. **Both heads are linear readouts
over $\phi$** — `GroupSum` is just the special case with fixed block-structured
$\{0,1\}$ weights. So the model is

> **[argmax over $C$ linear forms] ∘ [layered fan-in-$r$ LUT circuit].**

Because the domain $\{0,1\}^n$ is **finite**, "universal approximation" is not a
density statement — it is *exact representation*: can the architecture compute
**every** classifier $g:\{0,1\}^n\to\{1,\dots,C\}$?

## The one governing quantity: PTF degree

Every node of arity $r$ is a multilinear polynomial of degree $\le r$. Composing
$D$ layers multiplies degree: a depth-$D$ body produces features of degree
$\le r^{D}$. A linear head takes the sign of a linear form in those features.
Hence, for binary classification, the whole model computes exactly a

$$\boxed{\;\textbf{degree-}r^{D}\textbf{ polynomial threshold function (PTF).}\;}$$

(For $C>2$ classes, each pairwise decision boundary is such a PTF.) This single
statement organizes everything:

| body depth $D$ | representable class | universal? |
|---|---|---|
| $1$ | degree-$r$ PTFs | **no** |
| $\lceil\log_r n\rceil$ | degree-$n$ PTFs = **all** functions | **yes** |
| general $D$ | degree-$r^{D}$ PTFs | iff $r^{D}\ge n$ |

- **Tree/body depth** sets the achievable degree $r^{D}$ — the real lever for
  universality.
- **The head** (GroupSum *or* linear) supplies the outer threshold; it sets
  *which* degree-$r^{D}$ PTF and how finely weighted, **not** the degree. Swapping
  GroupSum for a learned linear decoder changes trainability and weight
  resolution, **not** the representable class.
- **Width** controls how much of the degree-$r^{D}$ class you actually reach.

## One layer is *not* universal (any width)

At $D=1$ the features are $r$-juntas (degree $\le r$), so one layer + head realizes
exactly the **degree-$r$ PTFs**. Width $\to\infty$ only refines the weights — it
cannot raise the degree. The clean witnesses are parities, which have PTF-degree
exactly $k$ (Minsky–Papert):

| single layer | can compute | **cannot** compute (any width) |
|---|---|---|
| arity 2 (`gate16`) | all degree-2 PTFs | **XOR of 3 inputs** (degree 3) |
| arity 6 (LUT-6) | all degree-6 PTFs | **XOR of 7 inputs** (degree 7) |

So an arity-2 `LogicNet` with `depth=1` and a billion nodes still cannot learn
3-bit parity. This is the point where the textbook one-hidden-layer UAT analogy
breaks: an MLP hidden unit reads *all* $n$ inputs (effectively arity $n$); a logic
node is capped at arity $r$, hence degree $r$.

## Universality: depth $\log_r n$, width up to $2^{n}$

Raise the depth to $D=\lceil\log_r n\rceil$. Now the body can compute **min-terms**
— a full conjunction $\bigwedge_i \ell_i(x)$ of all $n$ literals is an associative
AND, hence an arity-$r$ tree of depth $\lceil\log_r n\rceil$. Exactly one min-term
fires at any input, so a linear/`GroupSum` head over the min-terms assigns an
independent score to every point of the cube — **any** $g$, exactly:

- put each class-$c$ point's min-term in block $c$ (GroupSum), or give it weight
  $1$ in row $c$ (linear decoder); the unique firing min-term picks the class.

This is the universal construction. Its cost:

- **body depth** $\lceil\log_r n\rceil$ (just deep enough for one conjunction),
- **width** up to $2^{n}$ min-terms (Shannon–Lupanov sharing brings the worst-case
  node count down to $\Theta(2^{n}/n)$; see below),
- the **head** does the rest.

So the *base universal classifier* is **log-depth, growing-width** — the intuition
that "one shallow layer, width $\to\infty$" almost works is right once the depth is
$\log_r n$ rather than $1$, because the head is an unbounded-fan-in aggregator.

### Body depth vs. total gate depth — where the hardness lives

The log-depth body does **not** mean the whole computation is shallow. A
`GroupSum`/linear head over $W$ features is an **unbounded-fan-in linear
threshold**: expanded into gates, its popcount + argmax has depth
$\Theta(\log W)=\Theta(n)$. The head absorbs the deep "OR-of-min-terms" that a
bounded-fan-in body would need $\Theta(n/\log_2 r)$ layers to do itself. Two
different depth accountings, both true:

| what is counted | universal depth |
|---|---|
| **LUT-tree body layers** (before the head) | $\lceil\log_r n\rceil$ |
| **total logic-gate depth**, expanding the head's popcount | $\Theta(n)$ |
| **pure fan-in-$r$ circuit**, if the final decision must also be one logic node | $\Theta(n/\log_2 r)$ (formula depth) |

The library's own FPGA export shows this directly: the gate fabric collapses to
~2 LUT levels while the **GroupSum popcount is the critical path (~28 of 30
levels)** — the logic is shallow, the head is deep.

### GroupSum vs. a learned linear head

Because both are linear readouts, they represent the **same** class at a given
body. The differences are practical:

- a learned linear head with **real** weights needs multiply–accumulates — it
  breaks the "pure Boolean, no multiplies" property `GroupSum` preserves (a
  popcount). The `"ternary"` {class}`~silogic.heads.LearnedDecoder` recovers it
  with $\{-1,0,+1\}$ weights (a signed popcount);
- a linear head can up-weight informative trees, so it typically needs **fewer**
  trees than `GroupSum` for the same accuracy — a width/trainability win, not an
  expressiveness one.

## Size: the Shannon–Lupanov floor

Depth $\log_r n$ makes the model universal; **width** is what a worst-case target
costs. A layered arity-$r$ net with $N=W\!\cdot\!D$ nodes over $M\le N+n$ signals
realizes $\le\big(2^{2^{r}}M^{r}\big)^{N}$ circuits, and there are $C^{2^{n}}$
targets, forcing

$$N\bigl(2^{r}+r\log_2 M\bigr)\ \gtrsim\ 2^{n}\log_2 C
\qquad\Longrightarrow\qquad
N\ \gtrsim\ \frac{2^{n}\log_2 C}{2^{r}+r\,n}.$$

This is tight: Lupanov's construction matches it, and for fan-in 2 the constant is
exactly $1$ — almost every single-output $g$ needs $(1\pm o(1))\,2^{n}/n$ gates and
none fewer. Note $C$ enters only as $\log_2 C$ (the class indicators share
sub-circuitry).

For *structured* targets the width collapses far below $2^{n}/n$: any symmetric
function (depends only on the popcount) is a shallow adder tree, any degree-$r$
PTF is a single wide layer, any linearly separable target is `depth=1, width=C`.

## Arity 2 vs. arity 6

Both arities are universal — arity changes cost, not capability:

- **Depth.** An arity-$r$ LUT absorbs $\log_2 r$ levels of 2-input gates, so the
  universal body depth $\log_r n$ is $\log_2 r$ times smaller: arity 6 is
  $\approx\log_2 6\approx 2.585\times$ shallower than arity 2.
- **Size.** The bound above improves by the factor $2^{r}+rn$ in the denominator —
  in the $n$-dominated regime a constant $\sim r/2$ fewer nodes for arity 6, at the
  price of $2^{6}=64$ learnable truth-table bits per node vs. $2^{2}=4$ (a larger
  per-node soft→hard gap; see the arity warning in the [guide](guide.md)).
- **Degree reach per layer.** One layer reaches degree $r$: arity 6 clears 3-bit
  *and* up-to-6-bit parities that no arity-2 layer can, but still misses 7-bit
  parity — universality still needs depth.

## Scope and caveats

- **Representation, not learning.** These are statements about what
  `forward_hard` *can* compute, not about SGD finding it or about generalization.
- **Exact, because the domain is finite.** On $\{0,1\}^n$ a universal architecture
  represents every classifier exactly; "approximation" returns only if the inputs
  are a sub-sampled/continuous set (e.g. more thermometer bits approximating a
  real-valued feature).
- **Learned vs. random trees.** The degree-$r^{D}$-PTF ceiling is the same either
  way, but *learning* the trees (feature learning) drastically lowers the width
  needed for structured targets versus fixed/random candidate wiring (a
  random-feature kernel). This is why `connectome="topk"`/`"dense"` (learnable
  wiring) beat `"fixed"` at equal width.
- **`linear` / low-degree `polynomial` nodes are weaker.** They realize only
  threshold / bounded-degree functions per node and cannot represent XOR even at
  arity $r$; universality needs a full-LUT (`multilinear`/`hybrid`/`walsh`) or the
  16-gate (`gate16`) node.
