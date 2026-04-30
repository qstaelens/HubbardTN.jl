# HubbardTN

Contains code for constructing and solving general one-dimensional Hubbard models using matrix product states. The framework is built upon [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) and [TensorKit.jl](https://github.com/jutho/TensorKit.jl). The functionalities include constructing a Hubbard Hamiltonian with arbitrary interactions represented by the tensor U<sub>ijkl</sub>, as well as enabling hopping and interactions beyond nearest neighbors. Additionally, the framework supports U(1) and SU(2) symmetries for both particle and spin symmetry. Check out the examples for concrete use-cases. More information can be found in the [documentation](https://daanvrancken.github.io/HubbardTN.jl/stable).

To add this package to your Julia environment, do
```
julia> using Pkg
julia> Pkg.add("HubbardTN")
```
after which it can be used by loading 
```
julia> using HubbardTN
```
