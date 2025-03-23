#pragma once
#include "Eigen/Dense"
#include "types.hpp"
#include <stdint.h>
#include <vector>

// Forward declaration of Optimizer class
class Optimizer;

class Network {
  public:
    // Initializes a new Network from the given parameters.
    Network(uint32_t layers, std::vector<uint32_t> layer_sizes,
            std::vector<float (*)(float)> activation_funcs,
            std::vector<float (*)(float)> activation_derivatives,
            OutputActivationFunc output_activation, CostFunc cost,
            OutputLayerDerivative output_layer_derivative);

    // Feeds the given input through the network and returns the output.
    EigenVec feed_forward(EigenVecRef input) const;

    // Equivalent to feed_forward but updates the activations.
    EigenVec feed_forward_train(EigenVecRef input);

    // Trains the network using a single input-output pair.
    // Returns the cost before training.
    float train(EigenVecRef input, EigenVecRef output, Optimizer& optimizer);

    // Getters
    uint32_t get_layers() const;
    std::vector<uint32_t> get_layer_sizes() const;
    const std::vector<EigenMat>& get_weights() const;
    const std::vector<EigenVec>& get_biases() const;
    std::vector<float (*)(float)> get_activation_funcs() const;
    std::vector<float (*)(float)> get_activation_derivatives() const;
    OutputActivationFunc get_output_activation() const;
    CostFunc get_cost() const;
    OutputLayerDerivative get_output_layer_derivative() const;
    const std::vector<EigenVec>& get_activations() const;

  private:
    // Helper for feed_forward functions
    EigenVec feed_forward_helper(EigenVecRef input, bool cache) const;

    // Number of layers in newtork, including input, hidden, and output layers.
    uint32_t layers;
    // The size of each layer.
    std::vector<uint32_t> layer_sizes;
    // The weights connecting each layer with the next.
    // weights[i] shape: (layer_sizes[i] cols, layer_sizes[i+1] rows).
    std::vector<EigenMat> weights;
    // The biases in each hidden layer.
    std::vector<EigenVec> biases;
    // The activation function used in each hidden layer.
    std::vector<float (*)(float)> activation_funcs;
    // Derivatives of the activation functions
    std::vector<float (*)(float)> activation_derivatives;
    // Activation function used in output layer.
    OutputActivationFunc output_activation;
    // Cost function for network: (predictions, real values) -> cost
    CostFunc cost;
    // Combined derivative of output activation and cost function
    OutputLayerDerivative output_layer_derivative;
    // Stores the activations at each layer when feeding forward in train mode.
    mutable std::vector<EigenVec> activations;
};