#include <Eigen/Core>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <nn.h>
#include <optimization.h>
#include <loss.h>

using namespace neural;

template <typename T, enum LOSS_TYPE l, enum ACT_TYPE a>
void
check_gradient(T &ref, loss_function<l,a> &loss, struct nn_layer &layer, Eigen::VectorXd &input, Eigen::VectorXd &des)
{
  Eigen::MatrixXd weights_copy = layer.weights;
  Eigen::VectorXd tmp(des.size());
  double neg = 0.;
  double pos = 0.;
  double deriv_numeric = 0.;
  double eps = 0.0001;
  // calculate gradient via finite differences 
  // and compare
  for (int r = 0; r < layer.weights.rows(); ++r)
    { 
      for (int c = 0; c < layer.weights.cols(); ++c)
        {
          layer.weights = weights_copy;
          layer.weights(r, c) -= eps;
          ref.forward_pass(input, tmp);
          neg = loss(tmp, des);
          //neg = 0.5 * (tmp - des).squaredNorm();
          layer.weights(r, c) += 2.*eps;
          ref.forward_pass(input, tmp);
          pos = loss(tmp, des);
          //pos = 0.5 * (tmp - des).squaredNorm();
          deriv_numeric = (pos - neg) / (2. * eps);
          EXPECT_NEAR(deriv_numeric, layer.weights_deriv(r,c), 1e-4);
        }
    }
  Eigen::VectorXd bias_copy = layer.bias;
  for (int b = 0; b < layer.bias.size(); ++b)
    {
      layer.bias = bias_copy;
      layer.bias(b) -= eps;
      ref.forward_pass(input, tmp);
      neg = loss(tmp, des);
      //neg = 0.5 * (tmp - des).squaredNorm();
      layer.bias(b) += 2.*eps;
      ref.forward_pass(input, tmp);
      pos = loss(tmp, des);
      //pos = 0.5 * (tmp - des).squaredNorm();
      deriv_numeric = (pos - neg) / (2. * eps);
      EXPECT_NEAR(deriv_numeric, layer.bias_deriv(b), 1e-4);
    }
}

TEST(nn, forward_pass)
{
  nn_layer layer(3, 4, LOGISTIC);
  Eigen::VectorXd input(3);
  Eigen::VectorXd output(4);
  Eigen::VectorXd des(4);
  // set weights
  for (int r = 0; r < 4; ++r)
    for (int i = 0; i < 3; ++i)
      {
        layer.weights(r,i) = 0.1 * (r*3+i);
      }

  input(0) = 0.5;
  input(1) = 0.3;
  input(2) = 0.2;

  des(0) = 0.5174929;
  des(1) = 0.5914590;
  des(2) = 0.6615032;
  des(3) = 0.7251195;
  // calculate forward pass
  layer.forward_pass(input, output);
  for (int r = 0; r < 4; ++r)
    {
      //printf("\n%f %f\n", output[r], des[r]);
      EXPECT_NEAR(output[r], des[r], 1e-4);
    }
  // and calculate again to make sure everything is cleaned up correctly
  layer.forward_pass(input, output);
  for (int r = 0; r < 4; ++r)
    {
      EXPECT_NEAR(output[r], des[r], 1e-4);
    }
}

TEST(nn, net_io)
{
  srand(time(0));
  std::vector<int> dimensions;
  std::vector<enum ACT_TYPE> activations;
  // prepare network topology
  dimensions.push_back(8);
  dimensions.push_back(5);
  dimensions.push_back(8);
  activations.push_back(RECT);
  activations.push_back(LOGISTIC);
  // allocate new net
  nn net(dimensions, activations);
  // init randomly
  net.initRandom();
  // save to file
  // get a temporary filename
  char ftemp[] = "/tmp/fileXXXXXX";
  int fd;
  fd = mkstemp(ftemp);
  // make sure we have a valid file descriptor
  EXPECT_TRUE(fd != -1);
  // init a string with the filename
  std::string fname(ftemp);
  EXPECT_TRUE(net.to_file(fname));
  // read copy
  nn net2;
  EXPECT_TRUE(net2.from_file(fname));
  // and copare
  EXPECT_EQ(net.ltype, net2.ltype);
  for (size_t l = 0; l < net.layers.size(); ++l)
    {
      EXPECT_EQ(net.layers[l].input_dim, net2.layers[l].input_dim);
      EXPECT_EQ(net.layers[l].output_dim, net2.layers[l].output_dim);
      EXPECT_EQ(net.layers[l].activation_type, net2.layers[l].activation_type);       
      EXPECT_NEAR((net.layers[l].bias - net2.layers[l].bias).norm(), 0., 1e-4);
      EXPECT_NEAR((net.layers[l].weights - net2.layers[l].weights).norm(), 0., 1e-4);
    }
}


TEST(nn, backward_pass)
{
  srand(time(0));
  nn_layer layer(3, 4, LOGISTIC);
  loss_function<SQR, LOGISTIC> loss;
  layer.initRandom();
  Eigen::VectorXd input(3);
  Eigen::VectorXd des(4);
  Eigen::VectorXd out(4);
  Eigen::VectorXd err(4);
  Eigen::VectorXd tmp(4);
  input.setRandom();
  des.setRandom();

  layer.bias.setRandom();
  // foward pass random signal
  layer.forward_pass(input, out);
  // backward pass
  err = out - des;
  layer.backward_pass(err, input, out);
  check_gradient<nn_layer, SQR, LOGISTIC>(layer, loss, layer, input, des); 
}

TEST(nn, backward_pass_rect)
{
  srand(time(0));
  nn_layer layer(3, 4, RECT);
  loss_function<SQR, RECT> loss;
  layer.initRandom();
  Eigen::VectorXd input(3);
  Eigen::VectorXd des(4);
  Eigen::VectorXd out(4);
  Eigen::VectorXd err(4);
  Eigen::VectorXd tmp(4);
  input.setRandom();
  des.setRandom();

  layer.bias.setRandom();
  // foward pass random signal
  layer.forward_pass(input, out);
  // backward pass
  err = out - des;
  layer.backward_pass(err, input, out);
  check_gradient<nn_layer, SQR, RECT>(layer, loss, layer, input, des); 
}


TEST(nn, net_backward_pass)
{
  srand(time(0));
  std::vector<int> dimensions;
 std::vector<enum ACT_TYPE> activations;
  Eigen::VectorXd in(8);
  Eigen::VectorXd desired(8);
  Eigen::VectorXd out(8);
  // prepare network topology
  dimensions.push_back(8);
  dimensions.push_back(5);
  dimensions.push_back(8);
  activations.push_back(LOGISTIC);
  activations.push_back(LOGISTIC);
  // prepare input and desired
  in.setRandom();
  desired.setRandom();
  // allocate new net
  nn net(dimensions, activations);
  // init randomly
  net.initRandom();
  // setup loss functions
  loss_function<SQR, LOGISTIC> lossrms(net);
  loss_function<CE, LOGISTIC> lossce(net);
  for (int e = 0; e < 10; ++e)
    {
      desired.setRandom();
      net.clearGrad();
      // and check gradients for mean squared error case
      // --> first propagate through net
      net.forward_pass(in, out);
      net.backward_pass(desired);
      // then calculate finite differences 
      for (size_t l = 0; l < net.layers.size(); ++l)
        {
          check_gradient<nn, SQR, LOGISTIC>(net, lossrms, net.layers[l], in, desired); 
        }
    }
  // next set the loss type to cross entropy
  net.setLoss(CE);
  // and check if the gradients still work out fine :)
  net.clearGrad();
  net.forward_pass(in, out);
  net.backward_pass(desired);
  for (size_t l = 0; l < net.layers.size(); ++l)
    {
      check_gradient<nn, CE, LOGISTIC>(net, lossce, net.layers[l], in, desired); 
    }
  // finally check that weight decay works
  net.setDecay(0.001);
  net.clearGrad();
  net.forward_pass(in, out);
  net.backward_pass(desired);
  for (size_t l = 0; l < net.layers.size(); ++l)
    {
      check_gradient<nn, CE, LOGISTIC>(net, lossce, net.layers[l], in, desired); 
    }
}


TEST(nn, rprop)
{
  std::vector<int> dims;
  std::vector<ACT_TYPE> activations;

  dims.push_back(8);
  dims.push_back(5);
  dims.push_back(8);

  activations.push_back(LOGISTIC);
  activations.push_back(LOGISTIC);

  // allocate nn
  struct nn net(dims, activations);
  struct rprop r(net);
  net.initRandom().setDecay(0.000001).setLoss(CE);

  // training patterns
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd> > train;
  for (int i = 0; i < 8; ++i)
    {
      Eigen::VectorXd tmp(8);
      tmp.setZero();
      tmp(i) = 1.;
      train.push_back(std::make_pair(tmp, tmp));
    }
  //loss_function<SQR, LOGISTIC> lossrms(net);
  Eigen::VectorXd out(8);
  //for (size_t e = 0; e < 100; ++e)
  double err = 1e7;
  int count = 0;
  loss_function<SQR, LOGISTIC> lossrms(net);
  while (err > 0.0022)
    {
      err = 0.;
      for (size_t i = 0; i < train.size(); ++i)
        {
          net.forward_pass(train[i].first, out);
          net.backward_pass(train[i].second);
          //net.update_vanilla(0.1);
          err += lossrms(out, train[i].second);
        }
      err /= train.size();
      r.update(net);
      ++count;
      //std::cout << err << std::endl;
    }
  //std::cout << count << std::endl;

  // predict
  for (size_t i = 0; i < train.size(); ++i)
    {
      net.forward_pass(train[i].first, out);
      EXPECT_NEAR((train[i].second - out).norm(), 0., 0.055);
    }
}

TEST(nn, sgd)
{
  std::vector<int> dims;
  std::vector<ACT_TYPE> activations;

  dims.push_back(8);
  dims.push_back(5);
  dims.push_back(8);

  activations.push_back(LOGISTIC);
  activations.push_back(LOGISTIC);

  // allocate nn
  struct nn net(dims, activations);
  struct sgd s;
  net.initRandom().setDecay(1e-8).setLoss(CE);

  // training patterns
  std::vector<Eigen::VectorXd > in;
  std::vector<Eigen::VectorXd > out;
  for (int i = 0; i < 8; ++i)
    {
      Eigen::VectorXd tmp(8);
      tmp.setZero();
      tmp(i) = 1.;
      in.push_back(tmp);
      out.push_back(tmp);
    }

  loss_function<SQR, LOGISTIC> loss(net);
  // initialize sgd
  s.initialize(net, loss, in, out);

  //loss_function<SQR, LOGISTIC> lossrms(net);
  Eigen::VectorXd pred(8);
  //for (size_t e = 0; e < 100; ++e)
  double err = 1e7;
  int count = 0;
  while (err > 0.0008)
    {
      err = 0.;
      //std::cout << "starting epoch:" << count << std::endl;
      // NOTE: usually we would select a random subset here!
      for (size_t i = 0; i < in.size(); ++i)
        {
          net.forward_pass(in[i], pred);
          net.backward_pass(out[i]);
          s.update(net);
          err += loss(pred, out[i]);
        }
      err /= in.size();
      ++count;
      //std::cout << err << std::endl;
    }
  //std::cout << count << std::endl;

  // predict
  for (size_t i = 0; i < in.size(); ++i)
    {
      net.forward_pass(in[i], pred);
      EXPECT_NEAR((out[i] - pred).norm(), 0., 0.08);
    }
}

TEST(nn, svm)
{
  // allocate svm
  // NOTE: use hinge or log loss here
  struct svm s(3, LOGL);
  struct sgd trainer;

  // also NOTE: in the case of an svm 
  //            initialization to 0 might be better than random
  //s.initRandom();

  // training patterns
  std::vector<Eigen::VectorXd > in;
 std::vector<double> labels;

  Eigen::VectorXd x(3);
  x.setZero();
  x(0) = 1;
  x(2) = 1;
  in.push_back(x);
  labels.push_back(1.);

  x.setZero();
  x(0) = -1;
  x(2) = -1;
  in.push_back(x);
  labels.push_back(-1.);

  loss_function<LOGL, LINEAR> loss(s);
  // initialize sgd
  trainer.initialize(s, loss, in, labels);

  Eigen::VectorXd pred(1);
  Eigen::VectorXd des(1);
  //for (size_t e = 0; e < 100; ++e)
  double err = 1e7;
  int count = 0;
  while (err > 0.0008 && count < 40)
    {
      err = 0.;
      //std::cout << "starting epoch:" << count << std::endl;
      // NOTE: usually we would select a random subset here!
      for (size_t i = 0; i < in.size(); ++i)
        {
          des(0) = labels[i];
          s.backward_pass(des);
          trainer.update(s);
        }
      for (size_t i = 0; i < in.size(); ++i)
        {
          des(0) = labels[i];
          pred(0) = s.predict(in[i]); 
          err += loss(pred, des);
        }
      err /= in.size();
      ++count;
      //std::cout << err << std::endl;
    }
  //std::cout << count << std::endl;

  // predict
  for (size_t i = 0; i < in.size(); ++i)
    {
      pred(0) = s.predict(in[i]);
      des(0) = labels[i];
      EXPECT_NEAR((des - pred).norm(), 0., 0.08);
    }
}


int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
