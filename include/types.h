#ifndef NN_TYPES_H_
#define NN_TYPES_H_

namespace neural {

  enum ACT_TYPE {LOGISTIC, TANH, LINEAR, SOFTABS, RECT};
  enum LOSS_TYPE {SQR, CE, HINGE, LOGL};

  struct nn_layer;
  struct nn;
  struct rprop;
}

#endif
