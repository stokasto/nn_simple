#ifndef OPTIMIZATION_NEURAL_H_
#define OPTIMIZATION_NEURAL_H_

#include <vector>
#include <Eigen/Core>
#include <iostream>

#include "nn.h"
#include "loss.h"

namespace neural {
  class rprop
  {
  private:
    static const double RPROP_POS;
    static const double RPROP_NEG;

    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd> > last_gradients;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd> > stepwidth;

  public:
    rprop(const struct nn &net);
  
    struct rprop &
      setStepwidth(double s);
    struct rprop &
      reset();

    void
      update(struct nn &net);
  };

  class sgd
  {
  private:
    double eta0;
    double t;
    double lambda;
    
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd> > param_copy;
    template <enum LOSS_TYPE l, enum ACT_TYPE a>
      double
      evalEta(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<Eigen::VectorXd> &ys, int start, int end, double eta);
    
  public:
    sgd(double _lambda = 1e-5);
    sgd(double eta, double _lambda);
    
    template <enum LOSS_TYPE l, enum ACT_TYPE a>
      void
      initialize(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<Eigen::VectorXd> &ys, int start = 0, int end = -1);

    template <enum LOSS_TYPE l, enum ACT_TYPE a>
      void
      initialize(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<double> &ys, int start = 0, int end = -1);

    void
      update(struct nn &net);
        
  };
  
  template <enum LOSS_TYPE l, enum ACT_TYPE a>
    double
    sgd::evalEta(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<Eigen::VectorXd> &ys, int start, int end, double eta)
  {
    if (param_copy.empty())
      param_copy.resize(net.layers.size());

    // copy network parameters
    for (size_t i = 0; i < net.layers.size(); ++i)
    {
      param_copy[i].first = net.layers[i].weights;
      param_copy[i].second = net.layers[i].bias;
    }

    // run through the data and do one update step
    Eigen::VectorXd pred;
    for (int i = start; i < end; ++i)
    {
      net.forward_pass(xs[i], pred);
      net.backward_pass(ys[i]);
      net.update(eta);
    }
    // now do that again and calculate the induced loss
    double lval = 0.;
    for (int i = start; i < end; ++i)
    {
      net.forward_pass(xs[i], pred);
      lval += loss(pred, ys[i]);
    }
    // calculate average over examples
    lval = lval / (end - start + 1);

    // and copy back the parameters
    for (size_t i = 0; i < net.layers.size(); ++i)
    {
      net.layers[i].weights = param_copy[i].first;
      net.layers[i].bias = param_copy[i].second;
    }

    return lval;
  }

  template <enum LOSS_TYPE l, enum ACT_TYPE a>
    void
    sgd::initialize(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<Eigen::VectorXd> &ys, int start, int end)
  {
    if (end < 0)
      end = xs.size();
    const double factor = 2.;
    double low_eta = 1.;
    double low_cost = evalEta(net, loss, xs, ys, start, end, low_eta);
    double high_eta = low_eta * factor;
    double high_cost = evalEta(net, loss, xs, ys, start, end, high_eta);
    if (low_cost < high_cost)
      while (low_cost < high_cost)
      {
        high_eta = low_eta;
        high_cost = low_cost;
        low_eta = high_eta / factor;
        low_cost = evalEta(net, loss, xs, ys, start, end, low_eta);
      }
    else if (high_cost < low_cost)
      while (high_cost < low_cost)
      {
        low_eta = high_eta;
        low_cost = high_cost;
        high_eta = low_eta * factor;
        high_cost = evalEta(net, loss, xs, ys, start, end, high_eta);
      }
    eta0 = low_eta;
    std::cout << "Using eta0 = " << eta0 << std::endl;
  }
  
  template <enum LOSS_TYPE l, enum ACT_TYPE a>
    void
    sgd::initialize(struct nn &net, struct loss_function<l, a> &loss, const std::vector<Eigen::VectorXd> &xs, const std::vector<double> &ys, int start, int end)
  {
    if (net.layers[net.layers.size()-1].output_dim != 1)
    {
      std::cerr << "Error net has wrong output dimension: " << net.layers[net.layers.size()-1].output_dim << 
        " when the required dimension in calculateEta0() was 1" << std::endl;
    }
    // wrap the required labels into 1d vectors
    std::vector<Eigen::VectorXd> ys_vec;
    Eigen::VectorXd tmp(1);
    for (size_t i = 0; i < ys.size(); ++i)
    {
      tmp(0) = ys[i];
      ys_vec.push_back(tmp);
    }
    initialize(net, loss, xs, ys_vec, 0, ys_vec.size());
  }

}
#endif
