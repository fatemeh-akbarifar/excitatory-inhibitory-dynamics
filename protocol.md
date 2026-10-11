# Feedback experiment

Unshifted sigmoid F(x)=1/(1+exp(−x)); E' = −E+F(12E−wI+P−2); I' = (−I+F(10E−3))/2. Dimensionless time and activity; no data fitting. Representative P=0,1,4; initial (0.1,0.1); horizon200, late window100–200. Grid: P=0:0.2:4, w=8,10,12, initial E=I=0.05,0.5,0.9. SciPy solve_ivp, rtol1e−8, atol1e−10, max_step0.05, outputs0.05. Unresolved runs extend once to400, analysed over200–400.

Near-steady: both late ranges <0.001 and final derivative norm <1e−4. Apparently periodic: at least five E peaks (prominence0.001), interval CV<0.05 and amplitude drift between late-window halves<5%. Otherwise unresolved. These are descriptive finite-window labels. Equilibria: 25 starting guesses, residual<1e−7, [0,1] bounds, deduplication1e−5; no completeness guarantee. Jacobian eigenvalues describe local stability.

Checks: exact uncoupled relaxation, central-difference Jacobian, half-step comparison. At P=0 and1, perturb E by0.02 at t100; compare evolution, allowing phase shifts for oscillatory activity. Recovery to the baseline trajectory within0.0002 must persist for10 time units. Autonomous smooth two-dimensional equations here do not support a deterministic chaotic attractor. They omit individual spikes, anatomy and plasticity.
