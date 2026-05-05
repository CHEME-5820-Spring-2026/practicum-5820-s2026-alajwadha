[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nX1j7wVK)
# Practicum-5820-S2026
Practicum problem for CHEME 5820 Spring 2026.

This practicum extends Lecture L15a / Lab L15b (Spiking Hebbian Memory Networks)
into the **Stochastic Attention** family of training-free generative models,
applied to MNIST. Students implement the H-Mem ↔ modern-Hopfield ↔ SA bridge,
build a Langevin sampler over a Hopfield memory, and add a `Vector{Bool}` mask
on the attention softmax to make generation class-conditional.

Open [`CHEME-5820-Practicum-S2026.ipynb`](CHEME-5820-Practicum-S2026.ipynb) and
run all cells. Three tasks, three discussion questions, hidden `@testset`.
