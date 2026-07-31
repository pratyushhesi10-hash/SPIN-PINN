## SpinCoat PINN Lab

This is an interactive tool for exploring a research question from spin-coating physics: if you only get to measure a thin film's thickness at a few sparse, noisy points in time, can you work backward and figure out how its viscosity and evaporation rate were changing throughout the process, even though you never measured those directly? The app lets you generate synthetic data where you know the real answer and see how well the model recovers it, or upload your own real thickness measurements and see what comes out. It's built on top of a research project at NC State's ORaCEL lab studying inverse parameter recovery for organic semiconductor spin coating, and it's meant to make that project's findings tangible and explorable rather than just something you read about in a writeup.

## The physics behind it

When you spin-coat a polymer solution, two things are happening to the film at once. Centrifugal force is flinging liquid outward and thinning it, and the solvent is evaporating. Both effects are captured in a single differential equation:

```
dĥ/dτ = -K̃(τ)·ĥ³ - Ẽ(τ),      K̃(τ) = (ω/ω_ref)² · Ψ(τ)
```

Here `ĥ(τ)` is the film thickness (rescaled to be dimensionless), `Ψ(τ)` is a hidden function related to how the viscosity is changing over time, and `Ẽ(τ)` is a hidden evaporation rate. The spin speed `ω` shows up explicitly and is something you actually know for any given run, which turns out to matter a lot, as I'll get to.

Only `ĥ` is something you can actually measure in a lab, typically via ellipsometry, and usually only at a handful of time points per run. `Ψ` and `Ẽ` are never measured directly — the whole point of this project is trying to recover them anyway.

## How the app trains the model

The tool uses a Physics-Informed Neural Network, or PINN, which is really just a neural network that's penalized not only for missing your actual data, but also for disobeying the governing equation above. Concretely, three small networks get trained together: one per spin run that predicts thickness, plus one shared network for `Ψ(τ)` and one shared network for `Ẽ(τ)` that both spin runs draw from. Because neural networks are differentiable, the app can compute the network's own derivative `dĥ/dτ` automatically and check it against what the physics equation says it should be, everywhere, not just at your sparse measured points. That's what lets the model fill in the gaps between your handful of real data points with something that's still physically sensible.

A few design choices are baked in for good reason. The thickness network is built so that its starting value is mathematically guaranteed to be correct, rather than something it merely tries to learn, since it's written as `ĥ(τ) = 1 - τ · softplus(NN(τ))`, which forces `ĥ(0) = 1` exactly no matter what the network's weights are. Viscosity and thickness are also forced to stay positive by construction, using functions like `softplus` and `exp`, since negative thickness or negative viscosity would be physically meaningless.

## Walking through the tabs

The app is organized so you'd naturally move through it left to right. The Physics tab is a live simulator where you can drag sliders for viscosity strength and decay, evaporation strength and decay, and the spin speeds of two runs, and immediately see what thickness curves and hidden `Ψ`/`Ẽ` functions those settings would produce. It's there to build intuition before you train anything.

The Data tab shows the sparse, noisy synthetic measurements generated from whatever physics you dialed in, along with the dense collocation points used to enforce the physics equation during training. Train is where you actually run the optimization and watch the data loss and physics loss curves drop over time. Results is where you see whether it worked: recovered `Ψ(τ)` and `Ẽ(τ)` plotted against the true curves you set, recovered thickness per run, and a panel showing what's called the combined ODE term, which I'll explain more below because it matters more than it might seem at first glance.

The Manual/CSV tab lets you bring in your own real thickness-vs-time data instead of synthetic data, either by pasting CSV text, uploading a file, or typing values into a table directly in the browser. And the Model tab is just a short written recap of the equation and the architecture, useful if you want a refresher without digging through code.

## Using your own data

If you want to load real measurements, the CSV needs at minimum a `t` column and an `h` column. You can optionally add a `run_id` if you have more than one spin run, an `rpm` column so the app knows the spin speed for each run, and an `h_true` column if you happen to have an independent thickness reference to compare against. Here's what that looks like for two runs:

```csv
run_id,t,h,rpm
0,0,1.000,3000
0,10,0.430,3000
0,20,0.180,3000
0,30,0.084,3000
1,0,1.000,4500
1,10,0.250,4500
1,20,0.070,4500
1,30,0.012,4500
```

If your thickness and time values aren't already in the app's rescaled dimensionless form, there's a normalization panel where you set `h_wet` (the thickness at the very start) and `t_ref` (a representative timescale for your process), and the app handles the rescaling for you so everything lines up the way the model expects.

One thing worth knowing in advance: if all your runs share the same spin speed, you'll see a warning, because having only one spin speed means there's no way for the model to separate viscosity from evaporation, for reasons that are really the heart of what this whole project is about.

## What the research actually found, and why it matters here

This is the part I'd encourage you to actually read before trusting any specific number the app gives you, because it's not a minor caveat — it's the central result of the research this tool is built on.

With data from a single spin run, it turns out to be mathematically impossible to separate the viscosity term from the evaporation term. At any given instant, the governing equation gives you exactly one constraint, but there are two unknown functions you're trying to pin down, and infinitely many combinations of them satisfy that one equation equally well. I confirmed this directly: the model can fit the thickness data closely and drive the physics residual essentially to zero, while the recovered viscosity term is still off by well over 100%, sometimes closer to 200-300%, even though the *combined* contribution of viscosity and evaporation to the equation is recovered accurately, typically within about 11% error. That combined-term accuracy despite the individual split being wrong is the actual signature of this problem, and it's why the Results tab shows that combined panel prominently instead of hiding it.

Since spin speed enters the equation in a known way, specifically as a square, using two runs at different spin speeds should in principle break that ambiguity, because now you have two constraints and still only two unknowns. I checked this directly by solving the underlying system algebraically using true, noise-free trajectories, bypassing the neural network entirely, and it worked essentially perfectly, recovering both hidden functions to well under 1% error. That confirms the two-run version of this problem really is solvable in principle.

What surprised me is that training the actual PINN on this same two-run setup still struggled to find that solution, sometimes performing worse than the single-run case. I dug into why, and traced it to how the physics equation's sensitivity to the viscosity term scales with thickness cubed. As the film dries and thickness shrinks toward roughly 5% of its starting value, that cubed term becomes on the order of ten thousand times smaller than it started, which means the gradient signal available to correct the network's viscosity prediction becomes vanishingly small right when the film is thin, essentially cutting off the optimizer's ability to learn from most of the later part of the process. A noise sweep I ran, testing measurement noise from half a percent up to five percent, showed the viscosity recovery error staying roughly flat across that whole range, which points toward the real bottleneck being about how little useful information about viscosity exists late in the process, more than being about noise itself. Related to that, I estimated where in time the two-run setup's actual leverage on viscosity is concentrated, and found that about 95% of it sits before the process is even 20% of the way through, after which evaporation dominates the physics so completely that viscosity's contribution becomes a small, easily-confounded detail.

I also tried replacing the fully flexible neural network for viscosity with a much simpler two-parameter exponential curve, on the theory that fewer unknowns might be easier to pin down. It correctly found the right general decaying shape, but the actual amplitude was still off by something like 150-200%, and that error also didn't improve with lower noise, reinforcing that this isn't a matter of the network being too flexible or the data being too noisy so much as the decay rate genuinely not being well constrained by thickness measurements alone.

So, practically, what this app can reliably recover from typical sparse thickness data is the thickness curve itself, quite accurately, the evaporation function, reasonably well, usually within around 12% error, and the general shape of the viscosity decay, qualitatively. What it can't reliably recover is the precise magnitude or decay rate of the viscosity term, because in the regime this model covers, evaporation dominates the physics so much that viscosity ends up being a small, hard-to-isolate perturbation on top of it. Breaking that limitation for real would likely require something outside what thickness measurements alone can provide, like an independent viscosity measurement to anchor the amplitude, a much bigger contrast between spin speeds than what's typically used, or additional measurements like in-situ concentration data during drying.

If you're using this app on your own data, the practical takeaways are to trust the combined ODE term panel more than the individual `Ψ` and `Ẽ` curves, to load runs with denser sampling early in the process if you care at all about the viscosity recovery specifically, since that's where nearly all the real information about it lives, and to treat a single-run dataset as informative for thickness and evaporation trends only, not for viscosity, since that separation simply isn't there to be found no matter how well the model trains.

## A couple of implementation details worth knowing

The training loop sums the data loss and physics loss across however many runs are loaded rather than averaging them, which works fine as long as you're using the default two runs, but means the effective weighting between data and physics loss would shift if someone loaded three or more runs through the CSV tab, since the total loss magnitude scales with the number of runs. It's a quick fix if you want to make the tool robust to more runs, dividing both accumulated losses by the run count before combining them, but it's not something that affects the two-run results discussed above.

A couple of physical assumptions are also baked in rather than derived: evaporation is assumed to behave the same regardless of spin speed, and the starting wet-film thickness is assumed independent of spin speed too, both of which are reasonable simplifications but worth knowing are assumptions rather than something the model proves. The underlying equation also assumes the fluid behaves in a simple, Newtonian way, which may not hold perfectly for real, more concentrated polymer solutions later in the drying process.

## Getting it running

Go to the website:https://spin-pinn.streamlit.app/

## Where this comes from

The physics underlying this tool traces back to the classical spin-coating theory of Emslie, Bonner, and Peck (1958), and the PINN framework itself follows the approach introduced by Raissi, Perdikaris, and Karniadakis (2019). Everything specific to viscosity/evaporation recovery, the identifiability findings, and the app itself came out of ongoing undergraduate research at NC State's ORaCEL lab.
