# Thermodynamic Models: Fundamentals and Computational Aspects - Exercise Report
Yifan Hao

08/25/2024

## Chemical reaction equilibrium
The algorithm to calculate reaction equilibrium is implemented in `ReactionSystem.py`. Taking apparent components as input, it first constructs a list of element balance group; based on the apparent components type as monomer or inert, it creates a list of true component by considering the dimers of self- and cross-association. It also creates the $A_{ij}$ , which is the stoichiometry matrix of the true components and element groups.

The other input is the reaction equilibrium constant coefficients of self-association dimerization reactions.  It 
generates the function to calculate equilibrium constant and dimer chemical potentials.

### Estimate $\lambda$ values by fixed $n_t$
The first algorithm implemented in this class is `ReactionSystem.estimate_lambdas_by_fixing_nt(nt, lambdas)`, this function can be called after temperature, pressure, and apparent component molar flow are setup. It takes inputs of the fixed total moles of true components, and the initial guess of the $\lambda$ values. The algorithm to minimize $Q$ function is implemented here. 

Test of this function is in `TestReactionSystem.test_solve_lambdas_and_nt()` in `test_day9.py`. In the test, $T=360 K$ and $P=2 atm$ is setup to duplicate results at the end. The printed output:
```
estimated lambdas=[-2.99465699 -3.53831473 -3.53284552 -3.58288624 -1.32175584] in 9 iterations
```
`TestReactionSystem.test_different_lambdas()` can be used to test different initial guess of $\lambda$s for the optional questions. Here lists a few of the different initial guesses and their impact on the convergence:

|$\lambda_1-\lambda_5$|number of iterations|
|---|---|
|-100|NA|
|-10|7|
|-5|6|
|-1|9|
|0|11|
|1|13|
|5|21|
|10|31|
|100|NA|

Overall, negative initial values for $\lambda$s takes less iterations to converge, and once they converge, the final estimated $\lambda$ values are the same. When the initial guesses are too big or too small, the algorithm failed to converge, either due to invalid math operation of `exp` on very large number, or the compositions calculated from such initial guesses are too small, and the hessian matrix is very ill-conditioned to move the Newton method.

### Solve reaction equilibrium using Newton method
The final Newton solver is implemented in `ReactionSystem.solve(t, p, zi, initial_nt, initial_lambdas)`. It calls `ReactionSystem.estimate_lambdas_by_fixing_nt(nt, lambdas)` first to estimate the $\lambda$ values. 
`TestReactionSystem.test_solve_lambdas_and_nt()` tested this algorithm using the same condition to duplicate the final results:
```
estimated lambdas=[-2.99465699 -3.53831473 -3.53284552 -3.58288624 -1.32175584] in 9 iterations
Newton solver converged in 3 iterations
xi:
{
  "A": 0.054371920681348475,
  "B": 0.031381252101277256,
  "C": 0.031554781288397886,
  "D": 0.030002460519717545,
  "I": 0.30878159806224936,
  "AA": 0.029749647635936863,
  "AB": 0.06487616844590846,
  "AC": 0.06483558483141763,
  "AD": 0.06519862916237511,
  "BB": 0.035369471293650055,
  "BC": 0.07069469148486564,
  "BD": 0.07109054365523017,
  "CC": 0.03532523403196577,
  "CD": 0.07104607260697371,
  "DD": 0.035721946161120004
}
nt=0.6477069813958342
```

This algorithm can be tested in more conditions in `TestReactionSystem.test_different_lambdas()`:
![total moles of true components vs temperature](plots/nt_vs_T.png)

This plot shows the moles of true components increases with temperature, which means higher temperature decompose the dimers to the monomers.

![total moles of true components vs pressure](plots/nt_vs_P.png)

On the other hand, higher pressure decrease the total moles of true components, indicating the promotion of dimerization according to the Le Chatelier's principle.