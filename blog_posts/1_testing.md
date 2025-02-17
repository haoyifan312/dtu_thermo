# Testing in Scientific Software Development

Whenever we write a computational program - or any programs - we need to ensure that the software behaves as expected. The only way to confirm this is by running the program and verifying the results.

As the program evolves over time, it becomes increasingly complex, with many components interacting with each other. You may add new features to components, or requirements may change to meet the needs of other dependent components. How can we ensure that the original functionality remains intact despite these changes?

These are two common challenges in software development, both of which can be addressed by incorporating automated tests. I strongly agree with the following bold statement, at least from a software developer's perspective:

> "Testing is not about finding bugs" - Tip 66 from *The Pragmatic Programmer*

## Unit Test

Unit tests focus on verifying the smallest components of a program, typically a `class` if you are writing object-oriented code. If you follow test-driven development (TDD) practices - which is highly recommended - you write your tests before implementing the actual code. This approach ensures that all code is covered by tests and leads to better software design.

To illustrate how I write unit tests, consider the example of modeling a reactive system of organic acids. These acids can self-associate to form dimers or cross-associate with other organic acids:

$A + A \leftrightarrow A_2$

$B + B \leftrightarrow B_2$

$A + B \leftrightarrow AB$



Given the equilibrium constant of self-association, the cross-association equilibrium constant can be approximated as $K_{AB}=2\sqrt{K_{A_2B_2}}$

I created a [`ReactionSystem` class](https://github.com/haoyifan312/dtu_thermo/blob/main/ReactionSystem.py) to automatically calculate all possible species in such a system when provided with a list of different organic acid compounds or non-reactive compounds as input.

The first step is to test the constructor of my `ReactionSystem` class, which takes a list of initial components as input. Since this setup is used in multiple tests, I refactored it into a helper function:

``` python
def _build_example_reaction_system(self):
    monomers = [Component(name, ComponentType.MONOMER, [(name, 1)])
                for name in ('A', 'B', 'C', 'D')]
    all_components = [*monomers, Component('I', ComponentType.INERT, [('I', 1)])]
    return ReactionSystem(all_components)
```

This function creates a system consisting of four organic acid monomers (A, B, C, D) and one inert component (I) that does not react. Each component is represented by a `Component` data class, which stores its name, type (monomer, dimer, or inert), and its elemental information.

``` python
def test_build_true_component(self):
    system = self._build_example_reaction_system()
    monomers = ['A', 'B', 'C', 'D']
    inters = ['I']
    dimers = [monomers[i] + monomers[j]
                for i in range(len(monomers))
                for j in range(i, len(monomers))]
    true_components = [*monomers, *inters, *dimers]

    # Verify that the class generates the expected species list
    self.assertTrue(sorted(true_components), sorted(system.true_component_names))   
    self.assertEqual(len(system.true_component_names), 15)


    # Verify that the element-stoichiometry matrix is correctly constructed
    expected_element_stoi_matrix = np.array([[1, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 1, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2],
                                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])                                                
    self.assertTrue(np.allclose(system._mbg_by_component_matrix, expected_element_stoi_matrix))
```
This test verifies that the `ReactionSystem` class correctly generates 15 species, including monomers, dimers, and inert species, from a manually defined list. Additionally, it checks that each species contains the correct elemental composition.

There are several other [unit tests](https://github.com/haoyifan312/dtu_thermo/blob/main/test/test_day9.py) for `ReactionSystem`, including tests for automatically computing equilibrium constants for all possible reactions. In total, there are 10 unit tests, all of which run in under one second.

I hope this example provides insight into how unit testing works. While simply printing results for verification may seem sufficient, it does not safeguard against future changes that might alter the behavior of your code. Instead, using `assert` statements helps explicitly verify expected behaviors and catch unintended changes early. 

Another significant benefit of unit tests is that they serve as excellent documentation. Often, instead of reading the documentation for unfamiliar code, I examine its unit tests to understand how it should be used. Unit tests provide a single source of truth for how the code behaves in execution.

*Test-Driven Development* by Kent Beck is next on my reading list. I plan to share more about TDD in a separate blog post after finishing the book.

## Learning Test

When we decide to use external libraries, we not only read the documentation but also write code to test the library’s interface. Instead of treating these tests as one-off tasks, why not integrate them into our automated testing suite? This way, they can serve both as example code for using the library in our context and as safeguards against changes in the external library's interface.

To illustrate the concept of a learning test, consider the [thermclc](https://github.com/haoyifan312/dtu_thermo/blob/main/thermclc.py) subroutines from the DTU thermodynamics course. These subroutines calculate fugacity coefficients using an Equation of State (EoS). To use them, you first call the `INDATA` function to initialize the system, and then the `THERMO` function to compute the fugacity coefficients at a given set of conditions. These functions are likely designed this way because they were translated from FORTRAN subroutines.

Here’s an example of the `INDATA` function:

```python
def INDATA(NCA,ICEQ,LIST):
    """
    NCA - number of components
    ICEQ - EoS model type
    LIST - list of id for the components
    """
```

And here’s the `THERMO` function:

```python
def THERMO(T, P, ZFEED, MT, ICON):    
    """
    T - temperature
    P - pressure
    ZFEED - total component composition
    MT - desired phase
    ICON - option code

    return fugacity coefficient and its derivatives, and phase root of result
    """
```

To improve the usability of these functions, I created a wrapper called [`thermclc_interface`](https://github.com/haoyifan312/dtu_thermo/blob/main/thermclc_interface.py) to represent a system of components as an object. This wrapper enhances the readability of the original subroutines. If I decide to replace the EoS library in the future, the new interface will likely differ. Instead of updating the entire codebase, I can simply modify the code under the wrapper, leaving the rest of the code unaffected.

Here’s an [example](https://github.com/haoyifan312/dtu_thermo/blob/main/test/test_day1.py) of how I use the wrapper to test a functionality of the external library:

``` python
def test_pressure_derivative(self):
    p_mpa = 2
    t = 160.0
    inflow_moles = np.array(list(example_7_component.values()))
    # construct an object representing a 7-component system using SRK EoS
    with init_system(example_7_component.keys(), 'SRK') as stream:
        # calculate at given condition and component composition
        flash_input = FlashInput(t, p_mpa, inflow_moles)
        props = stream.calc_properties(flash_input, desired_phase=PhaseEnum.LIQ)

        # numerically evaluate the pressure derivative
        pert = 1e-6
        new_p = p_mpa + pert
        new_prop = stream.calc_properties(FlashInput(t, new_p, inflow_moles), desired_phase=PhaseEnum.LIQ)
        numerical_der = (new_prop.phi - props.phi)/pert

        # Compare the numerical derivative with the analytical derivative from the library
        analytical_der = props.dphi_dp
        print(f'numerical der={numerical_der}; analytical der={analytical_der}')
        self.assertTrue(np.allclose(numerical_der, analytical_der))

```

By using a wrapper and implementing learning tests, we not only ensure that our code interfaces with external libraries correctly but also guard against potential future changes, all while improving code clarity and maintainability.


## Integration Test

## Conclusion