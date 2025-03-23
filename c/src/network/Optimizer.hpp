#pragma once
#include "Eigen/Dense"
#include "types.hpp"

// Base class for optimizers
class Optimizer {
public:
    // Updates the given weights and biases from their gradients.
    virtual void update(EigenMat &weights, EigenVec &biases,
                        const EigenMat &weight_gradients,
                        const EigenVec &bias_gradients,
                        const EigenVec &activations) = 0;
    virtual ~Optimizer() = default;
};