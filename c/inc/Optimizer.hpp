#pragma once
#include "Eigen/Dense"
#include "types.hpp"

// Base class for optimizers
class Optimizer {
  public:
    // Updates the given weights and biases from their gradients.
    virtual Optimizer(Network& network){
      network = std::make_shared<Network>(network);
    }
    virtual void update() = 0;
    virtual ~Optimizer() = default;

  private:
    std::shared_ptr<Network> network;
};
