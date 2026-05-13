println("
##############
#  One-Band  #
##############
")

tol = 1e-2


#############
# COLLINEAR #
#############

symm = SymmetryConfig(U1Irrep, Trivial, 2, 1//1)
model = HubbardParams([0.0, 1.0], [10.0])

J = [1.0 0.0; 0.0 1.0]
s = [0.5, -0.5]
smf = SpinMeanField(J, s)

calc = CalcConfig(symm, model, smf)

@testset "Collinear spin field" begin
    gs = compute_groundstate(calc; tol=tol/10)
    ψ₀ = gs["groundstate"]
    H = gs["ham"]
    E0 = expectation_value(ψ₀, H)
    E = sum(real(E0))/length(ψ₀)
    @test E ≈ -0.4595 atol=tol
    up, down = density_spin(ψ₀, calc)
    spins = (up - down)./2
    @test abs(spins[1]) > 0.2
    @test abs(spins[2]) > 0.2
    @test isapprox(spins[1], -spins[2], atol=tol)
end


################
# NONCOLLINEAR #
################

s = [0.5 0.0 0.0; 0.0 0.0 0.5]
smf = SpinMeanField(J, s)
calc = CalcConfig(symm, model, smf)

@testset "Noncollinear spin field" begin
    gs = compute_groundstate(calc; tol=tol/10)
    ψ₀ = gs["groundstate"]
    H = gs["ham"]
    E0 = expectation_value(ψ₀, H)
    E = sum(real(E0))/length(ψ₀)
    @test E ≈ -0.4163 atol=tol
    sx = Sx(U1Irrep, Trivial)
    sy = Sy(U1Irrep, Trivial)
    @test abs(expectation_value(ψ₀, (1,)=>sx)) > 0.2
    @test isapprox(expectation_value(ψ₀, (1,)=>sy), 0.0, atol=tol)
end