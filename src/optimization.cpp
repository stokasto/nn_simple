#include "optimization.h"

namespace neural {

  const double rprop::RPROP_POS = 1.2;
  const double rprop::RPROP_NEG = 0.5;

  rprop::rprop(const struct nn &net)
  {
    last_gradients.clear();
    stepwidth.clear();
    for (size_t l = 0; l < net.layers.size(); ++l)
      {
        last_gradients.push_back(std::make_pair(Eigen::MatrixXd(net.layers[l].output_dim, net.layers[l].input_dim), Eigen::VectorXd(net.layers[l].output_dim)));
        stepwidth.push_back(std::make_pair(Eigen::MatrixXd(net.layers[l].output_dim, net.layers[l].input_dim), Eigen::VectorXd(net.layers[l].output_dim)));
      }
    reset();
  }

  struct rprop &
  rprop::setStepwidth(double s)
  {
    for (size_t l = 0; l < stepwidth.size(); ++l)
      {
        stepwidth[l].first.setConstant(s);
      }
    return *this;
  }

  struct rprop &
  rprop::reset()
  {
    for (size_t l = 0; l < last_gradients.size(); ++l)
      {
        last_gradients[l].first.setZero();
        last_gradients[l].second.setZero();
        stepwidth[l].first.setConstant(0.1);
        stepwidth[l].second.setConstant(0.1);
      }
    return *this;
  }

  void
  rprop::update(struct nn &net)
  {
    //std::cout << "update_rprop" << std::endl;
    for (size_t l = 0; l < net.layers.size(); ++l)
      {
        Eigen::MatrixXd &weights = net.layers[l].weights;
        Eigen::MatrixXd &weights_deriv = net.layers[l].weights_deriv;
        Eigen::MatrixXd &lweights_deriv = last_gradients[l].first;
        Eigen::MatrixXd &weights_step = stepwidth[l].first;
        Eigen::VectorXd &bias = net.layers[l].bias;
        Eigen::VectorXd &bias_deriv = net.layers[l].bias_deriv;
        Eigen::VectorXd &lbias_deriv = last_gradients[l].second;
        Eigen::VectorXd &bias_step = stepwidth[l].second;

        for (int i = 0; i < weights.rows(); ++i)
          for (int j = 0; j < weights.cols(); ++j)
            {
              if (weights_deriv(i,j) == 0.)
                continue;
              double sign = (weights_deriv(i,j) > 0) ? -1. : 1.;
              // std::cout << "dweight (" << i << "," << j << ") = " << weights_deriv(i,j) << std::endl;
              if (weights_deriv(i,j) * lweights_deriv(i,j) < 0)
                {
                  weights_step(i,j) *= rprop::RPROP_NEG;
                }
              else
                {
                  weights_step(i,j) *= rprop::RPROP_POS;
                }
              weights(i,j) += sign * weights_step(i,j);
            }
        for (int i = 0; i < bias.size(); ++i)
          {
            if (bias_deriv(i) == 0.)
              continue;
            double sign = (bias_deriv(i) > 0) ? -1. : 1.;
            if (bias_deriv(i) * lbias_deriv(i) < 0)
              {
                bias_step(i) *= rprop::RPROP_NEG;
              }
            else
              {
                bias_step(i) *= rprop::RPROP_POS;
              }
            bias(i) += sign * bias_step(i);
          }
        //std::cout << weights_deriv << std::endl << std::endl;
        lweights_deriv = weights_deriv;
        //std::cout << lweights_deriv << std::endl;
        lbias_deriv = bias_deriv;
        net.layers[l].clearGrad();
      }
  }

  sgd::sgd(double _lambda)
  {
    lambda = _lambda;
    t = 0;
    eta0 = 0;
  }

  sgd::sgd(double eta, double _lambda)
  {
    lambda = _lambda;
    t = 0;
    eta0 = eta;
  }

  void
  sgd::update(struct nn &net)
  {
    double eta = eta0 / (1 + lambda * eta0 * t);
    //std::cout << "sgd: next eta = " << eta << std::endl;
    // forward in time one step
    t += 1;
    // and do the update
    net.update(eta);
  }

}
