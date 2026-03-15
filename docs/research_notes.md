# Research Notes

## Threat Model
- Adversary has access to privatized location traces.
- Attacks: home inference, trajectory reconstruction, clustering attacks.
- Defense: geo-indistinguishability via Planar Laplace noise.

## Evaluation Plan
- Sweep epsilon values: 0.1, 0.3, 0.5, 1.0, 2.0.
- Metrics: mean location error, max privacy radius, service accuracy.
- Compare service results between true and noisy locations.
- Include Bangalore live demo: mask current location from user-entered origin, destination remains unmasked.

## Future Extensions
- Trajectory-level privacy mechanisms.
- Adaptive epsilon based on context.
- Privacy budget management.
- ML inference attack simulations.
