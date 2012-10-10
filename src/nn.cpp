#include "nn.h"

namespace neural {

  nn_layer::nn_layer(int idim, int odim, enum ACT_TYPE type)
  {
    reset(idim, odim, type);
  }

  struct nn_layer &
  nn_layer::reset(int idim, int odim, enum ACT_TYPE type)
  {
    input_dim = idim;
    output_dim = odim;
    decay = 0.;
    bias.setZero(odim);
    weights.setZero(odim, idim);
    bias_deriv.setZero(odim);
    weights_deriv.setZero(odim, idim);
    activation_type = type;
    // this can be set to update the bias more conservatively
    bias_supression = false;
    return *this;
  }
  
  struct nn_layer &
  nn_layer::initRandom()
  {
    static boost::mt19937 rng; // We could set a seed here
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
    bias.setZero(output_dim);
    for (int n = 0; n < weights.rows(); ++n)
      for (int w = 0; w < weights.cols(); ++w)
        {
          weights(n, w) = var_nor();
        }
    return *this;
  }

  struct nn_layer &
  nn_layer::setDecay(double _decay)
  {
    decay = _decay;
  }
  
  void
  nn_layer::clearGrad()
  {
    weights_deriv.setZero();
    bias_deriv.setZero();
  }

  void
  nn_layer::forward_pass(const Eigen::VectorXd &x, Eigen::VectorXd &y)
  {
    y = weights * x + bias;
    for (int i = 0; i < output_dim; ++i)
      {
        switch (activation_type)
          {
          case LOGISTIC:
            y(i) = 1. / (1. + std::exp(-y(i)));
            break;
          case TANH:
            y(i) = std::tanh(y(i));
            break;
          case RECT:
            for (int i = 0; i < y.size(); ++i) 
              {
                if (y(i) < 0.) 
                  {
                    y(i) = 0.;
                  }
              }
            break;
          case LINEAR:
          default:
            break;
          }
      }
  }

  void 
  nn_layer::calc_derivative(Eigen::VectorXd &err, const Eigen::VectorXd &x, const Eigen::VectorXd &activation, const Eigen::VectorXd dEdxi)
  {
    weights_deriv += dEdxi * x.transpose();
    weights_deriv += decay * weights;
    bias_deriv += dEdxi;
    err = weights.transpose() * dEdxi;
  }
  
  void
  nn_layer::backward_pass(Eigen::VectorXd &err, const Eigen::VectorXd &x, const Eigen::VectorXd &activation)
  {
    Eigen::VectorXd tmp_act;
    Eigen::VectorXd dEdxi;
    switch (activation_type)
      {
      case LOGISTIC:
        dEdxi = err.array() * activation.array() * (1. - activation.array());
        break;
      case TANH:
        dEdxi = err.array() * (1. + activation.array()) * (1. - activation.array());
        break;
      case RECT:
        tmp_act = weights * x + bias;
        dEdxi.resize(tmp_act.size());
        for (int i = 0; i < tmp_act.size(); ++i) 
          {
            if (tmp_act(i) < 0.)
              dEdxi(i) = 0.;
            else
              dEdxi(i) = err(i);
          }
        break;
      case LINEAR:
      default:
        dEdxi = err;
        break;
      }
    calc_derivative(err, x, activation, dEdxi);
  }

  void
  nn_layer::update(const double lrate)
  {
    weights -= lrate * weights_deriv;
    if (bias_supression)
      bias -= lrate * 0.01 * bias_deriv;
    else
      bias -= lrate * bias_deriv;
  }

  void
  nn_layer::update(const std::pair<Eigen::MatrixXd, Eigen::VectorXd> &delta)
  {
    weights -= delta.first;
    bias -= delta.second;
  }

  bool
  nn_layer::to_stream(std::ofstream &out)
  {
    int atype_i = (int)activation_type;
    out.write((char *) (&input_dim), sizeof(int));
    out.write((char *) (&output_dim), sizeof(int));
    out.write((char *) (&atype_i), sizeof(int));
    out.write((char *) (&decay), sizeof(double));
    out.write((char *) (&bias_supression), sizeof(bool));

    for (int i = 0; i < bias.size(); ++i)
      out.write((char*) (&(bias(i))), sizeof(double));
    for (int i = 0; i < weights.size(); ++i)
      out.write((char*) (&(weights(i))), sizeof(double));
    return true;
  }

  bool 
  nn_layer::from_stream(std::ifstream &in) 
  {
    // assume idim, odim and activation type are already set via constructor
    assert(input_dim > 0 && output_dim > 0);

    in.read((char*) (&decay), sizeof(double));
    in.read((char*) (&bias_supression), sizeof(bool));

    for (int i = 0; i < bias.size(); ++i)
      in.read((char*) (&(bias(i))), sizeof(double));
    for (int i = 0; i < weights.size(); ++i)
      in.read((char*) (&(weights(i))), sizeof(double));
    return true;
  }

  nn::nn()
  {
    ltype = SQR;
  }

  nn::nn(std::vector<int> &dimensions, std::vector<enum ACT_TYPE> &acttype)
  {
    assert(dimensions.size() == acttype.size() + 1);
    ltype = SQR;
    activations.push_back(Eigen::VectorXd(dimensions[0]));
    for (size_t l = 0; l < acttype.size(); ++l)
      {
        layers.push_back(nn_layer(dimensions[l], dimensions[l+1], acttype[l]));
        activations.push_back(Eigen::VectorXd(dimensions[l+1]));
      }
  }

  struct nn &
  nn::initRandom()
  {
    for (size_t l = 0; l < layers.size(); ++l)
      {
        layers[l].initRandom();
      }
    return *this;
  }

  struct nn &
  nn::setLoss(enum LOSS_TYPE type)
  {
    ltype = type;
    return *this;
  }

  struct nn &
  nn::setDecay(double d)
  {
    for (size_t l = 0; l < layers.size(); ++l)
      {
        layers[l].setDecay(d);
      }
    return *this;
  }
  
  void
  nn::forward_pass(const Eigen::VectorXd &x, Eigen::VectorXd &y)
  {
    activations[0] = x;
    for (size_t l = 0; l < layers.size(); ++l)
      {
        layers[l].forward_pass(activations[l], activations[l+1]);
      }
    y = activations[layers.size()];
  }

  void
  nn::backward_pass(const Eigen::VectorXd &desired_y)
  {
    Eigen::VectorXd tmp_err;
    // handle last layer first
    if (ltype == SQR)
      {
        tmp_err = activations[layers.size()] - desired_y;
        layers[layers.size()-1].backward_pass(tmp_err, activations[layers.size()-1], activations[layers.size()]);
      }
    else if (ltype == HINGE)
      {
        tmp_err.resize(desired_y.size());
        for (int i = 0; i < desired_y.size(); ++i)
          {
            const double z = desired_y(i) * activations[layers.size()](i);
            if ( z > 1)
              tmp_err(i) = 0;
            else
              tmp_err(i) = -desired_y(i);
          }
        layers[layers.size()-1].backward_pass(tmp_err, activations[layers.size()-1], activations[layers.size()]);
      }
    else if (ltype == LOGL)
      {
        tmp_err.resize(desired_y.size());
        for (int i = 0; i < desired_y.size(); ++i)
          {
            const double z = desired_y(i) * activations[layers.size()](i);
            // make sure we do not get inf here
            if (z > 18.)
              tmp_err(i) = -desired_y(i) * std::exp(-z);
            else if (z < -18.)
              tmp_err(i) = -desired_y(i);
            else
              tmp_err(i) = -desired_y(i) / (1. + std::exp(z));
          }
        layers[layers.size()-1].backward_pass(tmp_err, activations[layers.size()-1], activations[layers.size()]);
      }
    else if (ltype == CE)
      {
        Eigen::VectorXd dEdxi = activations[layers.size()] - desired_y;
        layers[layers.size()-1].calc_derivative(tmp_err, activations[layers.size()-1], activations[layers.size()], dEdxi);
      }
    else 
      {
        std::cerr << "Error: unknown loss type: " << ltype << std::endl;
        return;
      }
    // for all other layers
    for (size_t l = layers.size() - 1; l > 0; --l)
      {
        layers[l-1].backward_pass(tmp_err, activations[l-1], activations[l]);
      }
  }

  void
  nn::clearGrad()
  {
    for (size_t l = 0; l < layers.size(); ++l)
      {
        layers[l].clearGrad();
      }
  }

  void
  nn::update(const double lrate)
  {
    for (size_t l = 0; l < layers.size(); ++l)
      {
        /*
          std::cout << "lrate: " << lrate << std::endl;
          std::cout << "output: " << activations[l+1] << std::endl;
          std::cout << "updating with gradient: " << layers[l].weights_deriv << std::endl;
          std::cout << "bias_deriv: " << layers[l].bias_deriv << std::endl;
          std::cout << "weights: " << layers[l].weights << std::endl;
        */
        layers[l].update(lrate);
        layers[l].clearGrad();
      }
  }

  void
  nn::update(const std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd> > &deltas)
  {
    for (size_t l = 0; l < layers.size(); ++l)
      {
        layers[l].update(deltas[l]);
        layers[l].clearGrad();
      }
  }

  bool
  nn::to_stream(std::ofstream &out)
  {
    int ltype_i = (int)ltype;
    int num_layers = layers.size();
    bool res = true;
    out.write((char *) (&num_layers), sizeof(int));
    out.write((char *) (&ltype_i), sizeof(int));
    for (size_t l = 0; l < layers.size(); ++l)
      {
        res &= layers[l].to_stream(out);
      }
    return res;
  }

  bool
  nn::to_file(const std::string &fname)
  {
    bool res = true;
    std::ofstream out(fname.c_str(), std::ios_base::binary);
    res &= to_stream(out);
    out.close();
    return res;
  }

  bool 
  nn::from_stream(std::ifstream &in) 
  {
    int num_layers;       
    int ltype_i;
    in.read((char*) (&num_layers), sizeof(int));
    in.read((char*) (&ltype_i), sizeof(int));
    if (num_layers <= 0)
      return false;
    
    bool res = true;
    ltype = (LOSS_TYPE)ltype_i;
    for (int l = 0; l < num_layers; ++l)
      {
        int idim, odim, atype_i;
        in.read((char*) (&idim), sizeof(int));
        in.read((char*) (&odim), sizeof(int));
        in.read((char*) (&atype_i), sizeof(int));
        if (l == 0)
          activations.push_back(Eigen::VectorXd(idim));
        activations.push_back(Eigen::VectorXd(odim));
        layers.push_back(nn_layer(idim, odim, (ACT_TYPE)atype_i));
        res &= layers.back().from_stream(in);
      }
    return res;
  }

  bool
  nn::from_file(const std::string &fname) 
  {
    std::ifstream in;
    bool res = true;
    in.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
    
    try
      {
        in.open(fname.c_str(), std::ios_base::binary);
        res &= from_stream(in);
        
      }
    catch (std::ifstream::failure& e)
      {
        std::cerr << "Failed to load nn Model from file " << fname << std::endl;
        in.close();
        return false;
      }
    in.close();
    return res;    
  }

  svm::svm(int input_dim, enum LOSS_TYPE l, double lambda)
  {
    // default to log loss
    if (l != LOGL && l != HINGE)
      {
        std::cerr << "Warning: training svm with loss function other than HINGE or LOGL loss!" << std::endl;
      }
    ltype = l;
    activations.push_back(Eigen::VectorXd(input_dim));
    activations.push_back(Eigen::VectorXd(1));
    // create one neuron with linear activation
    layers.push_back(nn_layer(input_dim, 1, LINEAR));
    layers[0].setDecay(lambda);
    // update bias more conservatively
    layers[0].bias_supression = true;
  }

  int
  svm::predict(const Eigen::VectorXd &x)
  {
    int res;
    Eigen::VectorXd out(1);
    layers[0].forward_pass(x, out);
    // we only handle binary classification here
    res = (out(0) <= 0.) ? 1 : -1;
    return res;
  }


}
