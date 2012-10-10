#ifndef LOSS_NN_H_
#define LOSS_NN_H_

#include "types.h"
#include <algorithm>
#include <iostream>

namespace neural {

  template <enum LOSS_TYPE l, enum ACT_TYPE a>
    struct loss_function
    {
      const nn *net;

    loss_function() : net(0)
        {
        }
    loss_function(const nn &n) : net(&n)
        {
        }

      double
      operator()(const Eigen::VectorXd &y, const Eigen::VectorXd &des) const;
    };


  template <enum LOSS_TYPE l, enum ACT_TYPE a>
    double
    loss_function<l, a>::operator()(const Eigen::VectorXd &y, const Eigen::VectorXd &des) const
  {
    double res = 0.;
    if (l == SQR || a == LINEAR)
    {
      res = 0.5 * (y - des).squaredNorm();
    }
    else if (l == CE)
    {
      if (a == LOGISTIC)
	    {
	      for (int i = 0; i < y.size(); ++i)
        {
          // these are used for making sure that we do not calc log(0)
          // as we dont want inf and nan in our loss function :)
          const double tmp_add0 = (y(i) == 0.) ? 1. : 0.;
          const double tmp_add1 = (y(i) == 1.) ? 1. : 0.;
          res += des(i) * std::log(y(i) + tmp_add0)  + (1. - des(i)) * std::log(1. - y(i) + tmp_add1);
        }
	      res *= -1.;
	    }
      else 
	    { // TODO: CE loss for tanh ?
	      std::cerr << "ERROR: cannot compute CE  for activation " << a << std::endl;
	      res = 0.;
	    }
    }
    else if (l == HINGE)
    {
      // NOTE: we calculate the hinge loss element wise
      //       this should be the right thing to do if used
      //       for multiclass classification
      for (int i = 0; i < y.size(); ++i)
	    {
	      res += std::max(0., 1 - y(i) * des(i));
	    }
    }
    else if (l == LOGL)
    {
      for (int i = 0; i < y.size(); ++i)
	    {
	      const double z = y(i) * des(i);
	      // beware of too large results here
	      if (z > 18.)
          res += std::exp(-z);
	      else if (z < 18.)
          res += -z;
	      else
          res += std::log(1. + std::exp(-z));
	    }
    }
    else
    {
      std::cerr << "ERROR: unknown loss type: " << l << std::endl;
      res = 0.;
    }
    // add weight decay error term if net is available
    if (net)
    {
      for (size_t i = 0; i < net->layers.size(); ++i)
	    {
	      const nn_layer &layer = net->layers[i];
	      for (int r = 0; r < layer.weights.rows(); ++r)
        {
          for (int c = 0; c < layer.weights.cols(); ++c)
          {
            res += 0.5 * layer.decay * SQR(layer.weights(r, c));
          }
        }
	    }
    }
    return res;
  }
}
#endif
