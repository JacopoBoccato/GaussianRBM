module Gaussian_RBMs
using Random, LinearAlgebra, Statistics, Printf
using CairoMakie
using RestrictedBoltzmannMachines: sample_h_from_v
using AdvRBMs: calc_q, calc_Q
using Flux
using XLSX
using JLD2
using JLD2: @load, @save
using LIBSVM
using MLJ
using Optim
using DataFrames: DataFrame, nrow
using CSV
include("data_generate.jl")

end
