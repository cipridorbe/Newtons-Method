#include "network.hpp"
#include "Eigen/Dense"
#include "types.hpp"
#include <random>
#include <stdint.h>
#include <vector>

Network::Network(uint32_t layers, std::vector<uint32_t> layer_sizes,
                 std::vector<float (*)(float)> activation_funcs,
                 std::vector<float (*)(float)> activation_derivatives,
                 OutputActivationFunc output_activation, CostFunc cost,
                 OutputLayerDerivative output_layer_derivative) {
    assert(layers >= 2 && "network must have at least 2 layers");
    assert(layer_sizes.size() == layers &&
           "mismatch between layers and layer_sizes");
    for (int size : layer_sizes) {
        assert(size > 0 && "invalid layer size of 0");
    }
    assert(activation_funcs.size() == layers - 1 &&
           "mismatch between layers and activations");
    for (float (*activation)(float) : activation_funcs) {
        assert(activation != NULL && "invalid null activation");
    }
    assert(output_activation != NULL && "invalid null output activation");
    assert(cost != NULL && "invalid null cost function");
    assert(output_layer_derivative != NULL && "invalid null output derivative");
    this->layers = layers;
    this->layer_sizes = layer_sizes;
    this->activation_funcs = activation_funcs;
    this->activation_derivatives = activation_derivatives;
    this->output_activation = output_activation;
    this->cost = cost;
    this->output_layer_derivative = output_layer_derivative;
    this->activations.reserve(layers - 1);
    this->weights.reserve(layers - 1);
    this->biases.reserve(layers - 1);
    // Randomly init weights and biases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < layers - 1; i++) {
        EigenMat current_weights(layer_sizes[i + 1], layer_sizes[i]);
        EigenVec current_biases(layer_sizes[i + 1]);
        for (int col = 0; col < current_weights.cols(); col++) {
            for (int row = 0; row < current_weights.rows(); row++) {
                current_weights(row, col) = dis(gen);
            }
        }
        for (int row = 0; row < current_biases.size(); row++) {
            current_biases[row] = dis(gen);
        }
        this->weights.push_back(current_weights);
        this->biases.push_back(current_biases);
    }
}

EigenVec Network::feed_forward_helper(EigenVecRef input, bool cache) const {
    if (cache) {
        this->activations.clear();
    }
    EigenVec current = input;
    // Hidden layers
    for (int i = 0; i < this->layers - 1; i++) {
        current = this->weights[i] * current + this->biases[i];
        current.unaryExpr(this->activations[i]);
        if (cache) {
            this->activations.push_back(current);
        }
    }
    // Output layer
    this->output_activation(current);
    if (cache) {
        this->activations.push_back(current);
    }
    return current;
}

EigenVec Network::feed_forward(EigenVecRef input) const {
    return this->feed_forward_helper(input, false);
}

EigenVec Network::feed_forward_train(EigenVecRef input) {
    return this->feed_forward_helper(input, true);
}

// Getters
uint32_t Network::get_layers() const { return this->layers; }
std::vector<uint32_t> Network::get_layer_sizes() const {
    return this->layer_sizes;
}
const std::vector<EigenMat> &Network::get_weights() const {
    return this->weights;
}
const std::vector<EigenVec> &Network::get_biases() const {
    return this->biases;
}
std::vector<float (*)(float)> Network::get_activation_funcs() const {
    return this->activation_funcs;
}
std::vector<float (*)(float)> Network::get_activation_derivatives() const {
    return this->activation_derivatives;
}
OutputActivationFunc Network::get_output_activation() const {
    return this->output_activation;
}
CostFunc Network::get_cost() const { return this->cost; }
OutputLayerDerivative Network::get_output_layer_derivative() const {
    return this->output_layer_derivative;
}
const std::vector<EigenVec> &Network::get_activations() const {
    return this->activations;
}
