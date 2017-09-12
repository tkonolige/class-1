# Homework 0

- Forward difference formula for first derivatives.
- Backwards difference applied to forward difference for second derivatives.
- Tested with tanh.
- First order convergence on all first derivatives with any gridpoint spacing.
- Second order convergence on all second derivatives and gridpoint spacing except for arcsin spaced points. These points are clustered towards the middle. It appears the the method breaks down beacuse there aren't enough points near the endpoints.