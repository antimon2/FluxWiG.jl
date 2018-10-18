module FluxWiG

using Flux
using LinearAlgebra

export WiG

struct WiG{T<:AbstractFloat, L<:Union{Dense,Conv}}
    fn::L
end
(wig::WiG)(x::AbstractArray) = x .* σ.(wig.fn(x))

# Dense Constructor
function (::Type{WiG{T,Dense}})(n::Int, g::T=one(T), s::T=zero(T)) where {T<:AbstractFloat}
    Wg = Matrix{T}(I, n, n) .* g
    if s != 0
        Wg .+= randn(T, n, n) .* s
    end
    fn = Dense(param(Wg), param(zeros(T, n)))
    WiG{T,Dense}(fn)
end
(::Type{WiG})(fn::Dense{F,<:AbstractArray{T}}) where {F, T<:AbstractFloat} = WiG{T,Dense}(fn)

# Conv Constructor
function (::Type{WiG{T,Conv}})((fh, fw, n)::NTuple{3,Int}, g::T=one(T), s::T=zero(T)) where {T<:AbstractFloat}
    @assert isodd(fh) && isodd(fw) "Filter-size MUST be odd."
    padsize = (fh ÷ 2, fw ÷ 2)
    Wg = s == 0 ? zeros(T, fh, fw, n, n) : randn(T, fh, fw, n, n) .* s
    c0 = fh ÷ 2 + 1
    c1 = fw ÷ 2 + 1
    for ch in 1:n
        Wg[c0, c1, ch, ch] += g
    end
    fn = Conv(param(Wg), param(zeros(T, n)); pad=padsize)
    WiG{T,Conv}(fn)
end
(::Type{WiG})(fn::Conv{N,F,<:AbstractArray{T}}) where {N, F, T<:AbstractFloat} = WiG{T,Conv}(fn)

# params
# Flux.params(wig::WiG) = params(wig.fn)
Flux.@treelike WiG

end # module
