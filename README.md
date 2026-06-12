This code solves the mean-field Schrodinger Bridge Problem using a Sinkhorn Algorithm.
Run the mfsb_sinkhorn.m first then the particles_simulation.m.
mfsb_sinkhorn.m implements the generalized Sinkhorn Algorithm to obtain a universal control to steer particles under nonlocal interaction.
The result is a controller u_t(x) = \sigma \nabla \log \varphi(x), which is applied in particles_simulation.m. on 1000 particles (samples for the underlying interacting stochastic process).
ChatGpt has been used to assist write the local functions and organize this code. 
