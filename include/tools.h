#ifndef TOOLS_NH_H_
#define TOOLS_NH_H_

#include "types.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pthread.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

/**
 * authors: Tobias Springenberg, Jan Wuelfing, Manuel Blum
 */ 

namespace neural {

  /*
   * methods for plotting weights/features as 2-dimensional pictures
   */
  void
    featuresToImage(std::string filename, Eigen::MatrixXd &features, int px, int py, 
                    int num_channels);
  void
    featuresToImage(std::string filename, std::vector<Eigen::VectorXd> & features, 
                    int px, int py, int num_channels);

  /*
   * some useful preprocessing methods e.g. for whitening input patches
   * these methods will work in place and return a transformation matrix
   */
  Eigen::MatrixXd
    pca_whitening(std::vector<Eigen::VectorXd> & patches, int keep_components,
                  int drop_high, bool use_explained_variance, bool 
                  back_transform = false, int num_threads = 1);
  Eigen::MatrixXd
    zca_whitening(std::vector<Eigen::VectorXd> & patches, int num_threads = 1);
  // normalize a single input by mean subtraction and variance normalization
  void
    normalize(Eigen::VectorXd &patch);

  /*
   * methods for extracting small patches from an image
   * NOTE: the patches will be flattened to a vector
   */
  void
    extract_patches(IplImage *img, std::vector<Eigen::VectorXd> & patches, 
                    int patch_size_x, int patch_size_y, int patches_per_img);
 
}

#endif
