# NOVA-SIM — TX-34 Bipedal Robot Simulation Campaign

**Michel Maddalena · Synthar Core Systems · FVG, Italy · 2026**
**michel.m@syntharcoresystems.com · syntharcoresystems.com**

MuJoCo 3.x simulation campaign for the TX-34 NOVA bipedal robot.
66.23 kg · 25 DOF · Python 3.10 · ROS2-compatible URDF

> **The model walks.** It starts from rest in 2.4 s, walks forward for 120 s
> (654 support events, +2.5 m, reproduced on four MuJoCo builds and two operating
> systems) and stops standing. Five of six gait criteria met. What it does not do —
> carry a payload while walking, land a jump, walk sideways — is measured and
> declared below.

---

**Scope.** This repository contains the **locomotion and posture** simulation campaign only:
model, URDF, test modules and outputs. It is not the complete robot design, and subsystems
outside rigid-body locomotion are not included.

---

## What is this?

A virtual validation campaign for a bipedal robot platform. Every result is reproducible:
clone the repository, run one command, get the same number.

The model weighs 66.23 kg, stands 1.55 m, and has 25 actuated joints. Physics is
[MuJoCo](https://mujoco.org), the open-source engine developed by DeepMind.

---

## Requirements

> The walking canon is verified on **MuJoCo 3.1.6, 3.2.7, 3.10.0 and 3.11.0**, with **NumPy 1.26 and 2.4**, on Linux and Windows (see `requirements.txt`).

**Operating system** — Windows 10/11 · macOS 12+ · Ubuntu 20.04+

**Python 3.10**
- Windows: https://www.python.org/downloads/release/python-31011/ → *Windows installer
  (64-bit)*. During setup, check **"Add Python to PATH"**.
- macOS / Linux: https://www.python.org/downloads/

**Git**
- Windows: https://git-scm.com/download/win → install with default settings
- macOS: open Terminal and type `git --version` (installs on demand)
- Linux: `sudo apt install git`

**Python libraries** — installed in step 2 below: `mujoco`, `numpy`, `matplotlib`.

---

## Installation

### Step 1 — Download

Open a terminal (Windows: search for *Command Prompt* or *PowerShell*):

```bash
git clone https://github.com/SynthMike34/nova-sim
cd nova-sim
```

### Step 2 — Install the libraries

```bash
pip install -r requirements.txt   # mujoco>=3.1.6,<4 - numpy>=1.26 - matplotlib
```

Takes 2–5 minutes. It should end with `Successfully installed...`.

### Step 3 — Verify

```bash
python core/metriche_coppie.py --test
```

On Windows, if `python` is not recognised, use the `py` launcher instead — it is installed with
Python and does not depend on PATH:

```
py core\metriche_coppie.py --test
```

Expected first line:

```
[MASSA MODELLO] 66.228 kg
```

Every module prints the mass of the model it actually loaded, with an assert that stops
execution outside the 65.5–67.0 kg range. If you see `66.228 kg`, the installation is correct
and you are running the same model that produced the numbers below.

---

## How to run a simulation

Each module has two modes.

**Visual** — a window opens showing the robot. Press `ESC` to close.

```bash
python F1_statica/f1a_squat.py
```

**Test** — numbers only, no window. This is the mode used for verification.

```bash
python F1_statica/f1a_squat.py --test
```

Modules can be launched from any working directory: model paths are resolved relative to the
module file, and generated plots and metrics are written to `outputs/`.

Note: the visual mode needs a working OpenGL driver. On a headless machine or over a remote
session, use `--test`.

---

## Modules

### F1 — Static posture

| Module | What it does | Result |
|---|---|---|
| `F1_statica/f1a_squat.py` | Deep squat | za = 0.60 m |
| `F1_statica/f1b_reach.py` | Reach envelope | 0.48 / 0.43 / 0.47 / 0.47 m |
| `F1_statica/f1c_carico.py` | Load carrying | **2 kg** static, arm extended or tucked (elbow at its 8 N·m cap — C7 safeguard alignment); **walking payload retracted** — the canon gait falls under 100 g (see Key results) |
| `F1_statica/f1d_seduta.py` | Sit and stand up | 3.00 s · seat reaction 649 N · hip 73.4 Nm (size-36 foot) |

### F2 — Dynamic tests

| Module | What it does | Result |
|---|---|---|
| `F2_dinamica/t1_caduta.py` | Fall from 60 cm | 2.60 m/s · 425 ms early warning |
| `F2_dinamica/t12_hip_sway.py` | Lateral hip sway | 11.1° @ 0.9 Hz |
| `F2_dinamica/t18_power_loss.py` | Power loss strategies | coast is the least damaging |
| `F2_dinamica/tacchi_param.py` | Heel height 0–12 cm | cost is the contact width, not the height |

### F3 — Locomotion frontier

| Module | What it does | Result |
|---|---|---|
| `F3_frontiera/e1_capture.py` | Capture-point stepping | 6 support events · 2–4 cm placement error |
| `F3_frontiera/e2_timing.py` | Event-triggered timing | 32 synchronised cycles |
| `F3_frontiera/e3_accoppiato.py` | Commanded forward walking (historical bench, kp 600) | 11 support events · progress figure retired |
| `F3_frontiera/camminata_avanti.py` | **Forward walking - canon (L8, extension lever)** | **654 supports · 120 s alive · +6.44…+6.56 m on 4 builds** (this build +6.561833, three identical runs; stride 2.0 cm/cycle vs 0.72 without the lever). Lever-off base: +2.49 / +2.62 / +2.66 / +2.58 (max measured +9.6956, lever off) |
| `F3_frontiera/camminata_indietro.py` | Long mode (0.28 s support) | 420 supports · −25.93 m · 12.3 cm stride (roll −0.035). **Lever off, declared**: the extension lever is direction-specific — with it on, the backward island falls (12 supports, 5.7 s) |
| `F3_frontiera/partenza_arresto.py` | Start from rest / stop | first support 2.41 s · standing after stop · inertia −1.8…−3.6 cm by build (−1.76 on 3.11.0) |
| `F3_frontiera/e3_rms.py` | Hip roll thermal load | 53.7 Nm RMS in regime |
| `F3_frontiera/salto.py` | Jump | 270 ms flight · feet +6.0 cm · CoM +7.1 cm from take-off (B44, B49) |
| `F3_frontiera/e0_dita.py` | Toe mechanics | hallux extension at take-off: no effect on the size-36 foot — B45: 6 N·m cap, lever arm halved by the short forefoot; the +150% was the old 29.5 cm foot, retracted. *Measured before the B49 sign correction; the null result is unaffected.* |
| `F3_frontiera/e4_atterra.py` `e5_volano.py` `e6_supervisore.py` `e7_pipeline.py` | Landing pipeline stages | claim withdrawn — see Limitations |

### Core and tools

| Module | What it does |
|---|---|
| `core/gait_core.py` | Walking engine, used by the other modules |
| `core/metriche_coppie.py` | Torque metrics against the actuator ratings |
| `core/esporta_urdf.py` | URDF export for ROS2 |
| `tools/duty_anca_roll.py` | Hip roll duty cycle above nominal torque |
| `tools/confronta_range.py` | Compares model joint ranges against the software limits — reports **6 known discrepancies** (elbow sign convention, wrist, knee), listed in its own output |

---

## Key results @66.23 kg

| Result | Value | Class |
|---|---|---|
| Static balance envelope (fwd / back / lat) | 0.35 / 0.50 / 0.35 m/s on the size-36 foot (B46: the forefoot governs the front, −42%; the old 29.5 cm foot measured 0.60/0.50/0.40) | [C] |
| Squat depth | 0.60 m | [C] at software limits |
| Sit-to-stand | 3.00 s, feet-tucked strategy | [C] |
| Reach envelope (fwd / up / down / lat) | 0.48 / 0.43 / 0.47 / 0.47 m (`tx34_v1` model, 29.5 cm foot — the reach envelope is set by the arm and the CoM, not by the foot) | [C] |
| Forward walking (canon, L8) | **654 supports · +6.44…+6.56 m in 120 s across four builds** (this build +6.561833, three identical runs) · cycle unchanged · stride **2.0 cm/cycle vs 0.72** without the lever · leg-extension lever **EST_ZA 0.025** (cosine ramp over the last 40% of the true swing). The working band is **sign-consistent and value-irregular**: 0.028 → +3.50 m, 0.040 inverts the direction — the same statute as the roll plateau. The lever is direction-specific: the backward island runs with it off | [C] |
| Payload (static / walking) | **2 kg static** (`f1c_carico.py`, tx34_v1 model: extended, tucked and marching all cap at 2 kg — the limiter is the **elbow at 8 N·m** after the C7 safeguard alignment; the earlier 6 kg figure predates C7). **Walking: retracted** — the 2 kg figure was the old 29.5 cm foot. On the canon configuration (size-36 foot, E2 gait) **100 g on the torso brings it down** (12.9 s), 250 g in one hand halves its life, flip-flops end it in 6 s. The frontal hip is at its 80 N·m cap in every row — unloaded included — and no larger actuator fixes it (60→140 N·m tried). Standing it absorbs a 0.35 m/s push; walking it cannot carry 100 g: different limits — standing, the CoM sits over the support polygon; walking, it rides 8.1 cm outside it | [C] |
| Hip roll, postural thermal demand | 38.2 Nm RMS = 89% of nominal | [C] |
| Jump: flight time / clearance / CoM rise | 270 ms / +6.0 cm / +7.1 cm from take-off (PUNTA −0.35, size-36 foot) | [C] |
| Jump: first impact force | 2583 N = 398% of body weight (B47 criterion: first peak after take-off, not first peak inside the landing phase) | [C] |
| Hallux contribution to jump height | none on the size-36 foot (B45: 6 N·m cap, halved lever); the +150% was measured on the old 29.5 cm foot — retracted | [C] |
| E1 capture-point support events | 6 | [C] |
| E2 event-triggered cycles | 32 | [C] |
| E3 commanded forward walking | 11 support events (threshold ≥ 10; measured flight 15 ms over 11.2 s — no flight phase) | [C] |
| Fall: impact speed / impulse | 2.60 m/s · 274.9 N·s over 150 ms | [C] |
| Sit-to-stand: seat reaction | 649 N = 100% of weight (04/08 re-measure), Σ Fz balance closed | [C] |

`[C]` computed in simulation, reproducible with `--test`.

**Cross-platform reproducibility.** The torque metrics have been run on Linux (Python 3.12)
and on Windows 10 (Python 3.10) and agree **within 0.5%**, with an **identical sample count (2232)**. The residual
differences belong to the solver build, as with walking.

---

## Limitations

**No CAD.** Link masses and inertia tensors are design estimates from primitive decomposition.
Position-loop gains and upper-limb torque limits are placeholders. No thermal model, no
gearbox backlash, no joint compliance, no friction. Rigid contact with uncalibrated parameters.
Heuristic control throughout — no MPC, no learned policy, no torque control.

**No result has been validated on physical hardware.** These are reproducible orders of
magnitude, not certified measurements.

**Ten published figures have been withdrawn.** Each is listed below with the measurement that
replaced it.

| Claim | Published | What replaced it |
|---|---|---|
| Landing pipeline — time to upright rest | 0.36 s | not conserved at 66.23 kg |
| Toe-brake multiplier at touchdown | ×3.9 | not conserved at 66.23 kg |
| Hallux contribution to jump height | +150% | none, on a size-36 foot |
| Power loss — head velocity | 1.51 m/s | does not reproduce; 3.64 |
| Payload while walking | 2 kg | none — 100 g brings it down |
| Payload, static | 6 kg | 2 kg, elbow-limited after safeguard alignment |
| Forward balance envelope | 0.60 m/s | 0.35 on the anatomical foot |
| Cross-platform figures | identical to the last digit | agree within 0.5% |
| Jump — first impact force | 1840 N | **2583 N** — the module reported the second peak, inside the landing phase (B47) |
| Jump — flight / clearance / CoM rise | 250 ms · 8.2 cm · 5.3 cm | **270 · 6.0 · 7.1** — the ankle term had the wrong sign (B49) |

**Almost none of them were arithmetic errors.** They were correct numbers measuring the wrong
quantity: a window that included the robot standing still, a contact counted as a step, a
coefficient produced by reading the wrong actuator, a jump height measured from a height the robot
leaves before it jumps, an impulse anchored to a toe still on the ground. That failure mode is more
dangerous than a bug, because it looks like a result — and none of these was visible by reading the
code. They were found by instrumenting it.

**The last one is the most instructive.** A term in the jump controller was named *ankle push at
take-off* and was in fact dorsiflexion — it pointed the toes the wrong way. The published figure was
not a tuning compromise: it was an inverted sign. Correcting it raised the centre of mass 34% while
foot clearance fell, because the old term flicked the feet up without lifting the body.

The E4–E7 modules are kept in the repository because the code and the negative result are both part
of the record.


> **Canon C — the plateau (lever-off base).** Since L8 the walking canon includes the extension lever (see Key results); the figures below are the lever-off control, still reproduced bit-exact (this build +2.491220). The base gait is **654 supports, 120 s alive, +2.49 / +2.62 / +2.58 m on MuJoCo 3.1.6, 3.2.7 and 3.11.0, +2.66 on 3.10.0** (roll term −0.035, support imposed by clock at 0.180 s, leg kp ×2 = 400 with the feed-forward table left at ×3 — the quirk is part of the canon). Direction and order of magnitude agree across **four** solver builds (three identical runs per build) and across operating systems: on 3.10.0 the same run gives **+2.655481 m on Linux and +2.656692 m on Windows - one millimetre apart**. **+9.695622 m remains the measured maximum**, obtained at roll −0.05 on MuJoCo 3.11.0 + numpy 2.4.4 and declared for what it is: outside the −0.037…−0.034 plateau the direction of travel **alternates every ~3 thousandths of roll** (a comb, not a threshold) and depends on the numpy build as well. Event-triggered support (T_MAX 0.60) is *not* equivalent to the clock outside the campaign machine.
>
> **Model note.** `tx34_piedevero.xml` is the canon model **verbatim as measured**: it predates the safeguard torque alignment (C7) and actuator naming (B7) — realigning it would re-measure the canon. Its URDF carries the same pre-safeguard efforts. Declared, open item.
>
> **`gait_core` canary:** **(2, 0.330576)** with `tx34_v1` **as published** (C7, `waist_yaw` 2.5 N·m). The value **0.330795** is pre-C7 and is the one whose invariance was verified across three builds: **both are correct, each on its own model.**

**Sensitivity is high.** A 3.3% mass change reduced the achieved forward support-event count by 42% at
unchanged controller tuning. Any result here is conditional on a mass that has no CAD behind it.

---

## Key finding 1 — Mass sensitivity, separated by regime

Going from 62.8 to 66.23 kg (+5.5%):

| Regime | Effect |
|---|---|
| Ballistic (jump height) | **−54%** |
| Contact (landing pipeline) | **−100%**, withdrawn |
| Walking payload | **retracted** — see Key results |
| Quasi-static (balance envelope, reach) | **unchanged with mass** — but the balance envelope changed **−42% forward** with the foot geometry, see Key results |

The ballistic regime is mass-sensitive; the quasi-static support polygon is insensitive **to
mass**. It is not insensitive to its own geometry: shortening the forefoot from 22.5 to 16.5 cm
cost 42% of the forward balance envelope. The sensitivity also propagates into the thermal budget: the added mass required the
lateral lean to go from 0.12 to 0.15 rad to keep stability, and that lean generates the
postural component of the hip roll load below.

## Key finding 2 — The thermal cost of rigid contact

Hip roll actuator (MyActuator RMD-X8-P20, 43 N·m continuous rating), at real walking cadence
(T_sw 0.18 s), measured over the steady-state window:

| Component | Value | Share of cycle | Share of dissipation |
|---|---|---|---|
| Postural (< 70 N·m) | 38.2 N·m RMS = **89% of nominal** | 70.2% | 26% |
| Contact transients | ≈79 N·m, at the 80 N·m cap | **29.8%** | **64%** |
| **Total, continuous walking** | **53.7 N·m = 125% of nominal** | — | — |

> All thermal figures in this section are measured at leg `kp` = 200 [A] — the historical
> sample gait. The walking canon runs at `kp` 400 (C11); stiffness is not free.

**The transient peak is governed by position servo stiffness, not by gait kinematics.** With
`forcerange` raised to ±200 N·m the demand rises to 200 and saturates that limit too, and the
gait degenerates. The 80 N·m cap therefore acts as a design limiter, not as a measurement of a
requirement.

**Quantified target:** reducing the touchdown peak to **53 N·m** brings the total back to 100%
of nominal. The floor is the postural 89%: eliminating contact transients entirely would still
leave the actuator there, and the remaining lever is the lateral lean — a control strategy, not
an actuator size.

**Admissible duty:** walking up to roughly 61% of the time keeps the long-run RMS within
nominal. Indefinite continuous walking is not sustained. This is a prediction to be verified on
hardware with torque control.

---

## Retracted figures — the running list

Every number we have had to take back stays on the record. The full stories are in the
sections above; this table is the index, newest last.

| figure | was published as | now reads |
|---|---|---|
| Landing pipeline, time to upright rest | 0.36 s | withdrawn — did not survive the mass correction to 66.23 kg |
| Toe-brake multiplier | ×3.9 | withdrawn — same mass correction |
| Walking payload | 2 kg | 100 g brings the canon gait down — the 2 kg was the old 29.5 cm foot |
| Hallux contribution to jump height | +150% | none on the size-36 foot — the +150% was the old 29.5 cm foot |
| Power-loss head impact (coast) | 1.51 m/s (earlier canon, 29.5 cm foot) | 3.64 m/s on the anatomical foot — the 1.51 does not reproduce |
| Historical bench forward progress (e3, kp 600) | a progress figure | retired — support events only |
| Jump — flight / clearance / CoM rise | 250 ms · 8.2 cm · 5.3 cm | 270 ms · 6.0 cm · 7.1 cm — the ankle term had the wrong sign (B49) |
| Forward walking distance | +2.49 m in 120 s | **superseded, not wrong** — the leg-extension lever (EST_ZA 0.025) entered the canon: +6.44…+6.56 m across four builds. The old figure is the same gait with the lever off, and remains the base control, reproduced bit-exact |

The jump correction deserves one more line: the feet clearance goes **down** (8.2 → 6.0 cm) while
the centre-of-mass rise goes **up 34%** (5.3 → 7.1 cm). The old positive term was dorsiflexion —
it flicked the feet up without lifting the body. The number that matters is the CoM.

## Measurement conventions

- `[C]` computed in simulation — reproducible with `--test`
- `[A]` assumption to be verified on physical hardware
- `[PROPOSAL]` proposed design update, not implemented

**Mass sentinel.** Every module prints `[MASSA MODELLO] 66.228 kg` with an assert on the
65.5–67.0 kg range. No number in this campaign can exist without the mass that produced it —
the convention exists because an earlier batch ran at the wrong mass and the error was only
caught by comparison.

**Measurement window.** RMS and duty-cycle figures are computed over the steady-state window,
not the whole run: including the settling phase dilutes the denominator and returns optimistic
values. On the same data the hip roll RMS is 43.1 N·m over the full run and 53.7 N·m in regime.
Energy per support event, being an integral over completed support events, is insensitive to the window.

**Force balance.** Measured contact forces are reported with their closure check, Σ Fz against
m·g. A force that does not close is not published.

---

## Canon verification

Corpus consistency across documents, canon values and model is enforced by an internal
canon-audit tool with a mandatory negative self-test. It is a documentation-maintenance
instrument and is not distributed here.

Joint range consistency **is** verifiable in this repository:

```bash
python tools/confronta_range.py --test
```

It compares the ranges in the MuJoCo model against the software limits in
`config/joint_limits_v2.json`. The model previously ran with the hip 12° beyond the safeguard
threshold, which changed two published results — squat depth and sit-to-stand. Both were
re-measured and corrected.

---

## URDF for ROS2

```bash
python core/esporta_urdf.py
```

Generates `tx34_v1.urdf` (25 joints) and `tx34_v1_toes.urdf` (27 joints), ROS2 Humble and
later. The export is re-read and verified for total mass and joint count after generation: an
earlier version silently lost the 5.4 kg pelvis because a fixed joint was being merged into the
world body.

---

## Project structure

```
nova-sim/
├── models/          robot XML models and exported URDF
├── core/            walking engine, torque metrics, URDF export
├── F1_statica/      static posture tests
├── F2_dinamica/     dynamic tests
├── F3_frontiera/    locomotion frontier and thermal analysis
├── tools/           joint range comparison, duty cycle
├── config/          software joint limits
└── outputs/         generated plots (PNG)
```

---

## Open questions

Formulated as questions, not requests. If you have data on any of these, it would be useful.

1. Measured thermal derating for integrated planetary actuator modules held at 100–125% of
   continuous rating under a walking duty cycle. And how much does the rating fall inside an
   enclosed limb with no airflow?
2. With torque control or series elasticity at the ankle, does the touchdown peak fall below
   the 53 N·m that would bring the system back to nominal?
3. Is there a documented biped above 50 kg built on catalogue actuators, and how was
   frontal-plane hip torque handled?
4. Is the 2.9 ratio between event-triggered cycles and commanded walking consistent with what
   is observed before introducing predictive control, or is it specific to a heuristic
   controller?
5. What is the measured penalty of relocating ankle actuation proximally, at unchanged total
   mass? This is the measurement missing from this campaign.
6. Under power loss on non-backdrivable gearboxes, is there a passive strategy that satisfies
   both the settling-time and the head-velocity criterion? The measured answer here is no.

---

## Troubleshooting

**`python: command not found`**, or on Windows *"Python was not found; run without arguments
to install from the Microsoft Store"*

Python is installed but not on PATH — that message comes from the Microsoft Store alias, not
from Python. Use the `py` launcher, which is installed alongside Python:

```
py --version
py core\metriche_coppie.py --test
```

If `py` works, use it in place of `python` in every command on this page. Alternatively,
reinstall Python and check *Add Python to PATH*.

**`No module named 'mujoco'`**
Run `pip install mujoco`.

**`mujoco.FatalError: gladLoadGL error`** or the window fails to open
No OpenGL context available. Use `--test`, which needs no graphics.

**The window opens and closes immediately**
Normal in `--test` mode. Drop `--test` for the visual.

**`UnicodeEncodeError` when redirecting output to a file**
The Windows console code page is not UTF-8. Run:

```
set PYTHONUTF8=1
```

and repeat the command. If this is needed, please report it — the modules are meant to print
plain ASCII.

**`mj_name2id(..., mjOBJ_ACTUATOR, ...)` returns `-1` for every actuator name**

Intentional: actuators in `models/*.xml` carry no `name` attribute. Beware the silent
trap: numpy accepts `-1` as an index, so indexing actuator arrays with a stray `-1` reads
the **last** actuator of the list instead of raising an error - this has already produced
one clean-looking false positive during analysis. Resolve actuators through
`model.actuator_trnid` (joint -> actuator), as the `aid` dict in `core/gait_core.py` does:

```python
aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
       for i in range(m.nu)}
```

**A number does not match this README**
First check the first line of output: it must read `[MASSA MODELLO] 66.228 kg`. If it does and
the number still differs, please open an issue — a discrepancy on a different machine is
information worth having.

---

## Licence

MIT — free to use, modify and distribute with attribution.

## Contact

Michel Maddalena
michel.m@syntharcoresystems.com
syntharcoresystems.com
