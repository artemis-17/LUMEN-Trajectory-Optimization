# LUMEN — guided neural solar-sail trajectory optimization

This branch contains one population-based optimizer. It jointly searches departure date, departure-orbit phase, baseline steering parameters and a small residual neural network. It uses differential-evolution-style updates, not topology evolution or backpropagation.

The spacecraft gravity and sail equations are unchanged. A controller proposes cone/clock commands; numerical propagation determines the trajectory. A failed finite search is not a proof of infeasibility, and an approximate forecast is never counted as a completed flyby.

## Run, configure and view results

Run from this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py --mission mission_example.json --search search_settings.json --output mission_search --animate
```

There is no `--optimizer` flag anymore. All optimizer defaults are in `search_settings.json`; CLI values override only the options explicitly supplied:

```powershell
.\.venv\Scripts\python.exe main.py --population 12 --generations 100 --max-evaluations 240 --workers 4 --output another_run --animate
```

`--rounds` is an alias for `--generations`. These denote search rounds, not physical flight duration. `--max-evaluations` limits training propagations; one final exact reference propagation is additional. With the defaults, the maximum is 17 rounds, up to 238 training propagations, plus final validation. Cache reuse can reduce the actual count. This is an evaluation-count limit, not a wall-clock timeout.

The mission file controls the departure window, flight duration, ordered planet targets, sail properties and constraints. Your current `mission_example.json` target is preserved. Set both `epoch_start` and `epoch_end` to search a window; omit both for a fixed `epoch`. The `epoch` is also a reference seed when it lies in the window. Example overrides:

```powershell
.\.venv\Scripts\python.exe main.py --epoch-start 2034-01-01 --epoch-end 2040-12-31 --output date_window
```

Quick pipeline check, deliberately too short to complete its planetary mission:

```powershell
.\.venv\Scripts\python.exe main.py --mission examples/mission_quick.json --population 4 --generations 2 --max-evaluations 12 --workers 1 --output quick_run --animate
```

Open the run's `animation.html` in a browser and use Play or the timeline. To regenerate an animation without training:

```powershell
.\.venv\Scripts\python.exe main.py --replay mission_search/policy.pkl --animate
.\.venv\Scripts\python.exe main.py --replay mission_search/policy.pkl --full --animate
```

Replay uses the saved selected date, mission and controller, not the currently edited mission JSON. Without `--full`, it validates the saved trained horizon; `--full` attempts the entire requested flight duration. It can still take considerable time. The orange vector in the sail panel points toward the Sun; cyan is sail normal. Sail size and vector lengths in that panel are illustrative.

To seed a new search with an existing guided controller:

```powershell
.\.venv\Scripts\python.exe main.py --seed-policy previous_run/policy.pkl --output continued_search
```

This imports one candidate; it is not an exact resume of a previous population or random-number state. The seed's selected date must fall within the new window. Existing guided policy artifacts remain loadable, including older artifacts with retired settings. Pickle can execute code: load only trusted policies. Artifacts from the old algorithm should be used on their original branch.

## Project structure

| File | Responsibility |
| --- | --- |
| `main.py` | Single CLI for optimization and replay |
| `mission.py`, `mission_example.json` | Physical scenario, departure initial state and mission constraints |
| `search.py`, `search_settings.json` | Population/budget defaults and numerical accuracy profiles |
| `optimizer.py` | Candidate decoding, seeding, mutation, selection, verification and archives |
| `guided_policy.py` | Eight-feature, two-output residual neural network and steering parameters |
| `control.py` | Stateful departure/braking/energy/targeting guidance |
| `fitness.py` | Actual encounter assessment and search objective |
| `encounters.py` | Sun-only coast forecasts and Lambert shooting utilities |
| `rollout.py` | Connect controller, propagation, metrics and score |
| `dynamics.py`, `solar_sail.py` | Spacecraft force equations |
| `simulation.py`, `integrator.py` | Reference and persistent DOP853 propagation |
| `ephemerides.py` | SPICE, exact caches and guarded interpolation |
| `policy_io.py` | Guided-policy loading and trajectory export |
| `plotting.py`, `visualize_network.py` | Animation and residual-network diagram |
| `benchmark.py` | Numerical speed/error comparison utility |
| `tests/test_core.py` | Compact numerical/replay regression suite |

Saved runs and historical output directories are not deleted by this cleanup. The obsolete root genome/diagram are preserved locally in `archive/legacy_artifacts/`. The four scattered development test modules, obsolete training/configuration path, and dependency for the previous algorithm have been removed. The small `guided_policy.py` class/module name is intentionally retained so existing guided-policy pickles still load.

## 1. Physical scenario and propagation

Use SI units internally: position in meters, velocity in m/s, time in seconds, and gravitational parameters in m³/s². AU is 149,597,870,700 m; a year is 365.25 days. SPICE supplies geometric heliocentric J2000 states, with epochs expressed as TDB seconds past J2000. UTC strings are converted by SPICE.

For state x = [r, v], the unchanged equations are:

```text
dr/dt = v

dv/dt = -mu_sun * r / |r|³
        + sum_i mu_i * [(r_i-r)/|r_i-r|³ - r_i/|r_i|³]
        + a_sail

e_r = r/|r|
e_h = (r × v)/|r × v|
e_t = e_h × e_r

n = cos(alpha)*e_r + sin(alpha)*cos(delta)*e_t
                    + sin(alpha)*sin(delta)*e_h

a_sail = P_SRP*(A/m)*(AU/|r|)² *
         [(1-rho)*cos(alpha)*e_r + 2*rho*cos(alpha)²*n]
```

`P_SRP = 4.56e-6 N/m²`. Cone alpha is bounded to [0, pi/2]; clock delta is wrapped to [-pi, pi). Commands are held for `control_days`, normally one day. No impulsive maneuvers are added.

At each candidate departure epoch, the initial state is the Earth–Moon barycenter's heliocentric state plus a prescribed circular relative orbit. Its radius is `departure_radius_km` from that barycenter, not altitude. Speed is sqrt(mu_EMB / radius). `departure_phase_deg` rotates the relative position/tangent in a plane based on Earth's heliocentric radial/transverse/normal frame; `departure_inclination_deg` tilts that plane. Search varies phase, not radius, mass, area, reflectivity or an unreported departure delta-v.

Propagation stops on the first inward crossing of `solar_limit_au`, first outward crossing of `escape_radius_au`, or the current evaluation horizon. Perihelion roots refine the minimum solar-distance metric. The persistent DOP853 backend retains its adaptive step estimate between command intervals, bounds steps at command changes and refreshes its cached derivative afterward. The reference backend restarts `solve_ivp` at each command interval. Final exports use the reference backend.

## 2. Decision vector: 22 normalized variables

Every population member is a vector z in [0,1]^22. Decoding is fixed in `optimizer.decode`:

| Coordinates | Decoded quantity | Mapping/bounds |
| --- | --- | --- |
| 0 | Departure epoch | et_start + z0*(et_end-et_start), stored to UTC milliseconds |
| 1 | Relative departure-orbit phase | 360*z1 degrees |
| 2 | Osculating-perihelion switch q_switch | 0.12 + 0.22*z2 AU |
| 3 | Inbound/reset braking cone | 25 + 20*z3 degrees |
| 4 | Outer reset radius R_reset | 0.7 + 2.3*z4 AU |
| 5 | Targeting blend gain g | 3*z5 |
| 6–21 | Neural weights W, row-major | W = 1.2*(reshape(z[6:]) - 0.5), each weight in [-0.6, 0.6] |

The switch q_switch is a controller proxy computed from a Sun-only osculating orbit. It is not the actual minimum solar distance, and it may be below the thermal limit. Only the propagated thermal event determines whether a candidate violates the prescribed limit.

## 3. Controller and residual network

The controller carries two discrete states: its current guidance mode and next requested target index. It is not a recurrent neural network. Every rollout starts with mode `departure`, target index 0, and the same candidate-specific initial orbit.

Compute the Sun-only osculating quantities:

```text
h = r × v
e_vec = (v × h)/mu_sun - e_r
q_osc = |h|² / [mu_sun*(1 + |e_vec|)]
epsilon = |v|²/2 - mu_sun/|r|
```

Mode checks, in their implemented order:

| Mode/change | Condition/action |
| --- | --- |
| Start | `departure` |
| departure → inbound | Earth-barycenter separation exceeds 0.015 AU |
| inbound → energy | q_osc <= q_switch |
| energy → reset | epsilon < 0, radius > R_reset, and q_osc > 1.15*q_switch |
| reset → energy | q_osc <= q_switch or epsilon >= 0 |

Departure uses Earth-relative velocity to choose positive-work steering when possible. Inbound/reset use the candidate braking cone and retrograde clock pi. Energy mode favors positive heliocentric orbital work. Outer reset braking lowers perihelion again while the orbit is still bound, allowing repeated solar passages rather than merely spiraling toward a larger bound orbit.

For a desired work vector expressed in the radial/transverse/normal frame, the clock aligns with its transverse/normal components. Cone selection compares edge angles 0 and pi/2 with a pure-specular analytic angle and nearby offsets of -15, -7, 0, 7, 15 degrees, clipped into the cone bounds. Each candidate is scored using the actual reflectivity expression:

```text
work(alpha) = w_R*((1-rho)*cos(alpha) + 2*rho*cos(alpha)³)
              + |w_TN|*2*rho*cos(alpha)²*sin(alpha)
```

This is a small candidate-angle maximization, not an exact unconstrained optimal-control solution.

The residual network is y = tanh(W*f), with W of shape 2 by 8. Its feature order is:

| Feature | Value |
| --- | --- |
| bias | 1 |
| solar radius | clip(log(radius/AU), -3, 5) |
| radial speed | tanh(v_R / 30000 m/s) |
| solar energy | tanh(epsilon / 4.5e8 m²/s²) |
| target transverse | d_T / (AU + |d|) |
| target normal | d_N / (AU + |d|) |
| guidance mode | departure=-1, inbound=0, reset=0.5, energy=1 |
| Earth separation | D_E / (D_E + 0.01 AU) |

Here d is next-target position minus spacecraft position, resolved in the current sail frame. Target features are zero after all targets are passed.

Apply corrections of 12*y0 degrees to cone and 40*y1 degrees to clock, then clip/wrap. In energy mode inside R_reset, clock is restricted to [-pi/2, pi/2]. If radius < 1.08*r_min and radial velocity is negative, the command is overridden to face-on radial steering. These are command safeguards, not guarantees of thermal survival.

### Target steering and forecasts

Targeting is enabled only when g > 0, a target remains, radius > 0.5 AU, and the spacecraft is outbound or has positive solar energy. Plans refresh on target change or after:

```text
refresh_days = max(1, min(30, 14*(radius/AU)^1.5))
```

A Sun-only two-body coast is sampled at 24 future times over at most three years or the remaining full mission time. Future planet positions come from exact SPICE. A bounded scalar refinement searches the basin around the best coarse sample, with one-hour time tolerance and 24 iterations maximum. This forecast omits future sail thrust and planetary perturbations and can miss other minima.

The initial desired work direction is v/|v|. A future coast miss may add g*d_miss/(AU+|d_miss|), expressed in the sail frame. If a zero-revolution Sun-only Lambert plan is available, it supersedes this miss-direction blend:

```text
tof = min(2.5 years, max(2 days, current_target_distance / 20000 m/s),
          max(1 second, remaining_full_mission_time))

future_target = SPICE_target(epoch + time + tof)
v_L = Lambert departure velocity from current position to future_target
w_desired = v/|v| + g*(v_L-v)/(10000 m/s + |v_L-v|)
```

The controller only chooses sail angles from this desired direction. It never sets spacecraft velocity to v_L or applies that delta-v instantaneously. The cached plan is approximate and held until refreshed. Failed/degenerate Lambert solves fall back to coast/work steering. Recorded forecast misses come from the actual-state coast, not the hypothetical Lambert impulse.

The controller advances its target index only when a guidance-update observation is within `flyby_radius_au`. Assessment uses the finer saved history, so an encounter between command updates can be detected before the controller reacts.

## 4. Initialization and population update

Initial dates span the interval. The reference epoch replaces the first date if it is inside the interval. With at least three candidates, the second date is the start and the last is the end. Departure phases use `(reference_phase/360 + 0.381966*i) mod 1`.

Seeds cycle switch thresholds [0.18, 0.22, 0.26, 0.30, 0.16, 0.24] AU, use a 35.264-degree braking cone and 1 AU reset radius, and cycle targeting gains [0, 0, 0.5, 1.5, 0, 3]. Neural coordinates initially equal 0.5, meaning zero weights. Candidates with index >= 6 receive normalized-coordinate Gaussian perturbations with standard deviation 0.08. Odd-index candidates seed their clock/target-normal coordinate at 0.8, corresponding to weight 0.36. A warm-start policy replaces the first candidate.

For populations N >= 4, the standard proposal is:

```text
mutant_i = z_i + 0.6*(z_best-z_i) + 0.7*(z_a-z_b)
```

Indices a and b are distinct from one another and i. Add independent Gaussian jitter with standard deviation 0.025 to the 16 neural coordinates. Binomial crossover selects each mutant coordinate with probability 0.45, forcing at least one selected coordinate, then clips all coordinates to [0,1]. With N < 4, use full-vector Gaussian mutation with standard deviation 0.12 instead of the differential step.

Additional proposals, in order:

1. Every third non-transition round, randomize date and orbit phase of the worst slot.
2. During scouting, restore each slot's original date anchor, preserving date diversity.
3. During encounter/mission refinement, overwrite the first two trials with best-candidate copies shifted by +/- `180*0.45^k` days. With at least four candidates, overwrite trials 2 and 3 with +/- `0.12*0.7^k` normalized changes in the clock/target-normal weight. k counts targeting-refinement rounds across both phases.

Consequently, populations of four devote all later trials to coordinate refinement; a population such as 12 also retains global differential proposals. Small populations are mainly diagnostic. Clipping and fixed-date windows can create duplicate proposals; the cache can reuse them.

Each trial competes only with the screening score of its parent slot, replacing it when its score is greater or equal. Accurate verification does not itself confer a selection bonus. The accurately checked archive is separate from this population.

## 5. Rounds, horizons, verification and budget

Let N be population, G requested maximum rounds, B training-rollout budget, and V = min(2,N). The implemented round cap is:

```text
R = min(G, max(1, floor(B / (N+V))))
```

B must be at least N+1. If B cannot fund both verification choices in a small one-round run, only the choices fitting the cap execute. Every submitted propagation counts, including failures; cache hits do not. No extra rounds are added to consume budget saved by caching.

For full mission duration <= 4 years, every round uses that full horizon. Otherwise the horizon phases are scout=min(4 years,T), encounter=min(12 years,T), mission=T. One round uses scout only; two use scout then encounter. With at least three rounds, first and second counts are rounded 40% allocations, with at least one final mission round. Integer correction reserves that final round. With the defaults N=12, G=100, B=240, R=17 and the split is **7 scout, 7 encounter, 3 mission rounds**. For durations between 4 and 12 years the last two horizons are equal.

There are no hard advancement gates. At a phase transition, all surviving vectors are reevaluated at the new horizon/objective before further mutation; historical scores are not compared across phases.

After each screening/selection round, verify the current best screening candidate. If V=2, the other check is the runner-up on even zero-based rounds, or a seeded random non-best contender on odd rounds. Accurate checks use exact SPICE and original tolerances. Within a phase, archive a policy only when its accurate score strictly improves; ties retain the archive. Phase changes start a separate archive, preserving earlier phase files.

Cache keys include objective phase, horizon, fidelity and exact candidate bytes. A final additional reference rollout validates the latest phase's accurately checked champion. It is not necessarily the best policy for every earlier horizon, and finite-budget selection need not find the global optimum.

## 6. Exact objective and feasibility

All quantities below use the propagated trajectory, not planned Lambert arrivals. Let K be completed targets, M total targets, d the closest saved-sample distance to the first unfinished target, d_tol the requested flyby radius, r_f final solar distance, R_goal the outward milestone, q minimum solar distance, epsilon solar two-body energy, and nu the final excess-speed proxy in km/s:

```text
nu = sqrt(max(0, 2*epsilon_f)) / 1000

C = -100*log10(1 + d/d_tol), if K < M; otherwise 0
E = 10*tanh(nu/20) + 15*clip(r_f/R_goal, 0, 1)
A = 20 / (1 + abs(q/AU - 0.24)/0.15)
```

Scout score:

```text
J_scout = A + 15*tanh((epsilon_f-epsilon_0)/4.5e8) + E + 0.15*C
```

Encounter and mission score:

```text
J = C + (E if K=M else 0.25*E) + 300*K
    + 25*[all science complete and outward milestone reached]
    + [1000 + 50*(1-flight_days/(mission_years*365.25))]*[feasible]
```

Any thermal-limit termination overrides the score with **-10000**. Larger scores are better; negative values are normal. There is no artificial maximum of 2. Compare scores only within the same phase, horizon and requested mission.

Encounters are evaluated from saved samples in requested order. After a hit, the next target's search starts at a later sample. A final-sample hit cannot simultaneously count a later target. If no later samples exist, the next body's final-time distance is retained only as shaping information, never a completed encounter. Repeated samples inside tolerance are not a precision flyby/closest-approach certificate.

A reference-validated trajectory is `feasible_in_model` only when it:

- records every requested encounter in order;
- terminates at the outward radius before the current horizon ends;
- has positive solar two-body energy and outward radial velocity;
- meets `min_vinf_kms`.

Otherwise the report says `no_solution_found`, or `partial_horizon_candidate` if it did not attempt the full requested duration. A mission completed early can be successful even during a shorter search phase. `full_mission_evaluated` means the full horizon was attempted or every mission objective was completed early; it does not mean propagation always continued for every requested year.

Forecast summaries are diagnostic only. A perihelion proxy, planned transfer, thermal command safeguard or positive energy alone is not mission success or proof of infeasibility.

## 7. Numerical profiles and settings

`search_settings.json` contains only active defaults and accuracy controls:

| Setting | Default | Meaning |
| --- | --- | --- |
| population | 12 | Candidate slots |
| generations | 100 | Requested round cap |
| workers | 4 | Process workers, capped at population |
| max_evaluations | 240 | Training propagation cap |
| seed | 42 | Random seed |
| screen_rtol | 1e-6 | Screening relative tolerance away from close planetary guards |
| screen_position_atol_m | 100 | Screening absolute position tolerance |
| screen_velocity_atol_mps | 0.001 | Screening absolute velocity tolerance |
| screen_sample_hours | 24 | Screening history spacing is at least this value |
| screen_guard_au | 0.1 | Distance for restoring original solver tolerances |
| interpolate | true | Permit screening force-position interpolation |
| ephemeris_step_days | 0.5 | Hermite table sampling interval |
| ephemeris_guard_au | 0.05 | Minimum exact-force fallback distance |
| ephemeris_error_limit_m | 5000 | Maximum error allowed at interpolation audit samples |

Accurate and reference profiles use mission setup's `rtol=1e-10`, position absolute tolerance 0.1 m, velocity absolute tolerance 1e-6 m/s and maximum integration step one day. Screening spacing is max(mission sample spacing, screen_sample_hours). Final exports use min(mission sample spacing, one hour).

Screening force queries use a cubic Hermite table of exact positions/velocities covering the whole departure window plus the largest planned horizon, shared within each worker. It never extrapolates. Force evaluations revert to exact SPICE within max(ephemeris_guard_au, 3*flyby_radius_au) of any planet. At the start of persistent integration steps, the separate screen_guard_au check restores original tolerances nearby. This does not prove that every intermediate screening evaluation has full reference accuracy.

The interpolation audit checks quarter/mid/three-quarter points in up to 64 selected grid intervals against SPICE. It is a spot check, not a continuous error bound. Guidance/target queries, accurate archive checks and final reference validation always use exact SPICE. Process-local bounded caches reuse exact queries. Altering date changes the actual epoch; no time quantization or force interpolation is used in the final reference run.

## Outputs, compatibility and validation

Each run contains `search_request.json` (effective settings, versions and source hashes), `selected_mission.json`, `policy.pkl`, phase champions `policy_<phase>.pkl`, `network.npz`, `population.npz`, `training_history.json`, `trajectory.npz`, `report.json`, and optionally `animation.html` and `errors.json`.

`network.npz` stores W, feature names, baseline parameters and selected date/phase. The complete policy also needs the stateful controller implementation; use `policy.pkl` for normal replay. `population.npz` is an inspection snapshot, not an exact resumable optimizer checkpoint. `trajectory.npz` contains t in seconds, X with shape 6 by N in SI units, `[time, cone_rad, clock_rad]` commands, and recorded coast predictions/model tags.

The old test clutter is consolidated into one essential suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe benchmark.py --days 30 --policies 4 --repeats 2
.\.venv\Scripts\python.exe visualize_network.py --policy mission_search/policy.pkl --no-view
```

The suite checks date decoding, work steering, coast/Lambert consistency, persistent/reference agreement, science-priority scoring, budget enforcement and exact policy replay. Graphviz's system executable is optional for diagrams; absent it, the tool saves DOT source.

Earlier development validation reached 100 AU in about 27.8 years with a 22.5 km/s excess-speed proxy, but missed its requested planetary encounter. That is an escape baseline, not a completed science mission. Long searches, different targets and narrow encounters require additional numerical convergence and physical validation.

## Physical limits

The model uses planet-system barycenters, with no separate Moon. The example departure radius is 1,000,000 km from the Earth–Moon barycenter. It is not a validated Earth-centered/cislunar orbit. Planet surfaces, shadowing, thermal material properties and attitude/slew constraints are absent. Daily commands and sampled encounters can miss narrow events; independent close-approach refinement is required. Multi-target ordering in saved samples does not imply a dynamically favorable gravity assist.

The escape radius is a design milestone, not a modeled heliopause. The reported excess-speed proxy is osculating solar two-body energy converted to speed, not the exact ultimate asymptotic speed while SRP and planetary forces remain active. Zero-revolution, short-horizon Sun-only transfer plans can be poor predictors for distant targets; no target is guaranteed feasible.

Kernels in `kernels/` are DE440s, naif0012.tls and gm_de440.tpc. Mission/window endpoint coverage is checked before expensive evaluation. Direct runtime dependencies are listed in `requirements.txt`; numerical tests should be rerun after changing SciPy or the ephemerides.
