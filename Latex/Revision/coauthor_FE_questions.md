# JCOMA-25-4218 Revision 1 — Questions for the FEM co-author

We need precise numeric / methodological answers to address several reviewer comments on the FE model (Section 2.2) and the material card (Table 1). Six items, one short reply per question is sufficient.

---

## Q1 — Cohesive interface stiffness (Reviewer 2)

Table 1 currently lists:

| Parameter | Value as printed |
|-----------|------------------|
| Normal stiffness `K_nn` | **100 N/mm³** |
| Shear stiffness `K_ss = K_tt` | **35 N/mm³** |

Reviewer 2 argues these are three orders of magnitude too low — they should be **100 kN/mm³** and **35 kN/mm³** (≈ 10⁵ N/mm³ ; the physically realistic penalty stiffness for a toughened-epoxy adhesive).

**What value was actually used in the Abaqus .inp file?** A single-line answer is fine (e.g., "`100 000 N/mm³`" or "`100 N/mm³` — chosen as a soft penalty stiffness for a specific reason").

---

## Q2 — Cohesive fracture energies (Reviewer 2)

Table 1 lists `G_Ic = 400 J/m²` and `G_IIc = 600 J/m²`. The lamina fracture energies are in `N/mm`. Reviewer 2 asks for unit consistency. The straight conversion gives:

- 400 J/m² = **0.4 N/mm**
- 600 J/m² = **0.6 N/mm**

**Are these the values that were in the .inp file?** Confirm or correct.

---

## Q3 — Through-thickness ply discretisation (Reviewer 1)

Reviewer 1: *"Were the layers modelled explicitly in the base laminate and the repaired part?"*

The current text in §2.2 says SC8R elements were used, plus a passing mention of a "sub-laminate approach". That's ambiguous. Please confirm which of the following matches the actual model:

- **(a)** One SC8R element per ply — so 9 elements through the parent thickness (layup `[45/-45/90/0/90/0/90/-45/45]`) and 6 elements through the patch thickness (layup `[0/90/0/90/-45/45]`).
- **(b)** Sub-laminate stacks — plies grouped into N stacks, each stack one SC8R element with a composite section. If so, **what was the grouping**?
- **(c)** A single SC8R element through-thickness with a composite section integrating the full layup analytically.

---

## Q4 — Material identification (Reviewer 1)

Reviewer 1: *"Graphite/Epoxy UD IMS 24K 977-2 — there is not enough data to identify the material. Which IMS fibres? Is 977-2 an epoxy resin? State the manufacturer."*

My best guess from the literature is: **Toho Tenax IMS65 24K** carbon fibres impregnated with **Solvay/Cytec CYCOM 977-2** toughened epoxy resin, supplied as a unidirectional prepreg by Solvay. Please confirm — and correct any of the three pieces below:

- Fibre grade & manufacturer (e.g., Tenax IMS65 24K, Toho Tenax / Teijin Carbon)
- Resin grade & manufacturer (e.g., CYCOM 977-2, Solvay)
- Prepreg supplier / part number (e.g., Solvay CYCOM 977-2 UD prepreg)

---

## Q5 — Mesh convergence study (Reviewer 2 — Step #9 ahead)

Reviewer 2 raises the concern that the ~95 % strength-recovery limit may be **mesh-dependent**, because stress concentrations at the step corners are sensitive to element size. To answer this:

- Was a mesh convergence study performed? (The paper currently only says the cohesive zone length `L_cz` is spanned by 3–5 elements.)
- If yes: what element sizes were tested, and at what density was the result considered converged?
- Was the ~95 % saturation reproduced across multiple meshes (i.e., is the asymptote a stable feature, or could a finer mesh push it higher / lower)?

---

## Q6 — Anything else

Is there any other FE or material detail that should appear in §2.2 or Table 1 to make the model fully reproducible for a peer reviewer? (e.g., damage-evolution law type, BK exponent η for mixed-mode, viscous regularisation, etc.)

---

*Please reply inline under each question; one or two lines per item is plenty. I'll then incorporate the answers into the revised manuscript and Table 1.*
