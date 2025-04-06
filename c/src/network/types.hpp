#pragma once
#include "Eigen/Dense"

typedef Eigen::VectorXf EigenVec;
typedef Eigen::Ref<EigenVec> EigenVecRef;
typedef Eigen::MatrixXf EigenMat;
typedef Eigen::Ref<EigenMat> EigenMatRef;
typedef void (*OutputActivationFunc)(EigenVecRef);
typedef float (*CostFunc)(EigenVecRef predictions, EigenVecRef labels);
typedef EigenVec (*OutputLayerDerivative)(EigenVecRef predictions, EigenVecRef labels);