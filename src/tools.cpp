#include "tools.h"
#include "log_macros.h"

namespace neural {

  /*
   * some small helper methods
   */
  bool
  compare_pairs(std::pair<uint32_t, float> p1, std::pair<uint32_t, float> p2)
  {
    return (p1.second > p2.second);
  }

  // helper for computing the covariance in parallel
  struct CovarianceContainer
  {
    int start, end, id;
    std::vector<Eigen::VectorXd> * patches;
    Eigen::MatrixXd * cov;
    
  };

  static void*
  calcCovariance(void * arg)
  {
    struct CovarianceContainer cont;
    cont = *((struct CovarianceContainer *) arg);

    int log_step = (cont.end - cont.start) / 10;
    for (int k = cont.start; k < cont.end; ++k)
      {
        if ((k - cont.start) % log_step == 0)
          {
            LOG(0, << "Thread " << cont.id << ": " << k - cont.start << "/" << cont.end - cont.start);
          }

        for (int i = 0; i < cont.cov->rows(); ++i)
          {
            (*cont.cov)(i, i) += cont.patches->at(k)(i) * cont.patches->at(k)(i);
            for (int j = 0; j < i; ++j)
              {
                double q = cont.patches->at(k)(j) * cont.patches->at(k)(i);
                (*cont.cov)(j, i) += q;
                (*cont.cov)(i, j) += q;
              }
          }
      }
    pthread_exit(NULL);
  }

  void 
  extract_at_pos(IplImage *img, int channel, Eigen::VectorXd &v, int offset,
                 int next_y, int next_x, int patch_size_x, int patch_size_y)
  {
    LOG(10, "Image depth: " << img->depth);
    LOG(10, "Image Channels: " << img->nChannels);
    CvRect rect = cvGetImageROI(img);
    for (int i = 0; i < patch_size_y; ++i)
      {
        for (int j = 0; j < patch_size_x; ++j)

          {
            assert(rect.y + next_y + i < img->height);
            assert(rect.x + next_x + j < img->width);
            if (img->depth == 32)
              {
                v(offset + (i * patch_size_x + j))
                  = float(
                          CV_IMAGE_ELEM(img, float_t, rect.y + next_y + i, (rect.x + next_x + j) * img->nChannels + channel));
              }
            if (img->depth == 16)
              {
                v(offset + (i * patch_size_x + j))
                  = float(
                          CV_IMAGE_ELEM(img, uint16_t, rect.y + next_y + i, (rect.x + next_x + j) * img->nChannels + channel));
              }
            if (img->depth == 8)
              {
                v(offset + (i * patch_size_x + j))
                  = float(
                          CV_IMAGE_ELEM(img, uint8_t, rect.y + next_y + i, (rect.x + next_x + j) * img->nChannels + channel));
              }
          }
      }
  }
  

  /*
   * end helper methods
   */

  void
  featuresToImage(std::string filename, Eigen::MatrixXd &features, int px, int py, int num_channels) 
  {
    std::vector<Eigen::VectorXd> features_tmp;
    // assume features are stored row wise in the matrix
    for (int i = 0; i < features.rows(); ++i)
      features_tmp.push_back(features.row(i));
    featuresToImage(filename, features_tmp, px, py, num_channels);
  }

  void
  featuresToImage(std::string filename, std::vector<Eigen::VectorXd> & features, int px, int py, int num_channels)
  {
    std::cout << "Num. of Channels: " << num_channels << std::endl;

    // total number of features
    int num_features = features.size();

    // the space between to features
    int border = 5;

    // calculate grid dimension
    int num_feat_in_row = floor(sqrt(num_features));
    int row_counter = 0;

    // image dimensions
    int image_width = (px + border) * num_feat_in_row;
    int image_height = (py + border) * (num_features / num_feat_in_row + 1);

    // create new image
    IplImage * img = cvCreateImage(cvSize(image_width, image_height), IPL_DEPTH_8U, num_channels);
    cvSet(img, cvScalar(0));

    // first sort by variance of color channels
    std::vector<std::pair<int, float> > sorted;
    int dim_patch = px * py;
    for (int k = 0; k < (int) features.size(); k++)
      {
        Eigen::VectorXd patch_tmp(dim_patch);
        Eigen::VectorXd patch = features[k];
        float var = 0;
        double min = patch.minCoeff();
        patch.array() -= min;
        double max = patch.maxCoeff();
        if (max > 0)
          patch /= max;

        if (num_channels == 3)
          {
            for (int i = 0; i < dim_patch; ++i)
              {
                patch_tmp(i) = (0.3 * patch(i) + 0.59 * patch(dim_patch + i) + 0.11 * patch(
                                                                                            2 * dim_patch + i));
              }
          }
        else
          {
            patch_tmp = patch;
          }

        patch_tmp.array() -= patch_tmp.mean();
        var = patch_tmp.norm();
        sorted.push_back(std::make_pair<int, float>(k, var));
      }
    std::sort(sorted.begin(), sorted.end(), compare_pairs);

    for (int k = 0; k < num_features; ++k)
      {
        //      Eigen::VectorXd & feature = features[k];
        Eigen::VectorXd & feature = features[sorted[k].first];
        assert(px * py * num_channels == feature.size());

        // normalize feature
        double min = feature.minCoeff();
        feature.array() -= min;
        double max = feature.maxCoeff();
        if (max > 0)
          feature /= max;

        // std::cout <<  "Feature # " << k << std::endl;
        for (int x = 0; x < px; ++x)
          {
            for (int y = 0; y < py; ++y)
              {
                int pos_x = x + (k % num_feat_in_row) * (px + border);
                int pos_y = y + row_counter * (py + border);

                assert(pos_x >= 0 && pos_x < img->width);
                if (!(pos_y >= 0 && pos_y < img->height))
                  {
                    std::cout << pos_y << " " << img->height << std::endl;
                    assert(false);
                  }

                if (num_channels == 3)
                  {
                    CV_IMAGE_ELEM(img, uint8_t, pos_y, pos_x * img->nChannels + 2) = feature(
                                                                                             (y * px) + x) * 255;

                    CV_IMAGE_ELEM(img, uint8_t, pos_y, pos_x * img->nChannels + 1) = feature(
                                                                                             (y * px) + x + (px * py)) * 255;

                    CV_IMAGE_ELEM(img, uint8_t, pos_y, pos_x * img->nChannels + 0) = feature(
                                                                                             (y * px) + x + 2 * (px * py)) * 255;
                  }
                if (num_channels == 1)
                  {
                    CV_IMAGE_ELEM(img, uint8_t, pos_y, pos_x) = feature((y * px) + x) * 255;
                  }
              }
          }

        if ((k + 1) % num_feat_in_row == 0)
          {
            row_counter++;
            //std::cout <<  "Counting up a row to : " << row_counter;
          }
      }

    // write image to disk
    cvSaveImage(filename.c_str(), img);
    // and free memory
    cvReleaseImage(&img);
  }

  void
  normalize(Eigen::VectorXd &patch)
  {
    /* --> preprocessing <-- */
    // brightness normalization
    double mean = patch.sum() / patch.size();
    LOG(7, "patch sum " << patch.sum() << ", " << "size: " << patch.size());
    LOG(7, "mean " << mean);
    patch.array() -= mean;
    double var = patch.squaredNorm() / patch.size();
    LOG(7, "var " << var << " " << patch.rows());
    // contrast normalization
    if (var > 0.) // actually only var == 0. can occur
      patch /= sqrt(var);
  }


  Eigen::MatrixXd
  pca_whitening(std::vector<Eigen::VectorXd> & patches, int keep_components,
                int drop_high, bool use_explained_variance, bool 
                back_transform, int num_threads)
  {
    // if dataset is empty return
    if (patches.empty())
      return Eigen::MatrixXd();
    // input vector dimension
    size_t N = patches.front().size();
    // size of dataset
    double m = patches.size();
    // init mean vector
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(N);
    // init covariance matrix
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(N, N);
    // eigenvalue vector
    Eigen::VectorXd D;
    // matrix D^-0.5
    Eigen::MatrixXd SqrtD = Eigen::MatrixXd::Zero(N, N);
    // compute mean vector
    LOG(1, << "Computing mean...");
    for (std::vector<Eigen::VectorXd>::iterator i = patches.begin(); i != patches.end(); ++i)
      {
        normalize(*i);
        mean += *i;
      }
    LOG(10, "sum of all is: " << mean.transpose());
    mean /= m;
    // center data (do this also for test points)
    for (std::vector<Eigen::VectorXd>::iterator i = patches.begin(); i != patches.end(); ++i)
      {
        *i -= mean;
      }
    // compute covariance matrix
    if (num_threads > 1)
      {
        LOG(1, << "Computing covariance matrix using " << num_threads << " threads...");

        pthread_t *threads = new pthread_t[num_threads];
        CovarianceContainer *thread_args = new CovarianceContainer[num_threads];
        int rc;

        int step = patches.size() / num_threads;
        for (int i = 0; i < num_threads; ++i)
          {
            thread_args[i].id = i;
            thread_args[i].start = i * step;
            thread_args[i].end = thread_args[i].start + step;
            thread_args[i].patches = &patches;
            thread_args[i].cov = new Eigen::MatrixXd();
            thread_args[i].cov->setZero(N, N);
            rc = pthread_create(&threads[i], NULL, calcCovariance, (void *) &thread_args[i]);
            assert(0 == rc);
          }

        /* wait for all threads to complete */
        for (int i = 0; i < num_threads; ++i)
          {
            rc = pthread_join(threads[i], NULL);
            assert(0 == rc);
            cov += *(thread_args[i].cov);
            delete thread_args[i].cov;
          }
        delete [] threads;
        delete [] thread_args;
      }
    else
      {
        // NOT PARALLEL
        LOG(1, << "Computing covariance matrix using NO threads!");
        int counter = 0;
        for (std::vector<Eigen::VectorXd>::iterator k = patches.begin(); k != patches.end(); ++k)
          {
            if (counter % 1000 == 0)
              LOG(0, << "Patch Nr: " << counter << "/" << patches.size());
            counter++;
            for (size_t i = 0; i < N; ++i)
              {
                cov(i, i) += (*k)(i) * (*k)(i);
                for (size_t j = 0; j < i; ++j)
                  {
                    double q = (*k)(j) * (*k)(i);
                    cov(j, i) += q;
                    cov(i, j) += q;
                  }
              }
          }
      }

    LOG(1, "Max: " << cov.array().maxCoeff() << " min: " << cov.array().minCoeff() << ", mean " << cov.array().mean());
    cov /= m - 1;

    // whiten data
    // eigendecomposition
    LOG(1, << "Eigen decomposition...");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov);

    // compute SqrtD
    D = eigensolver.eigenvalues();

    // explained variance
    // 1) sum up all eigenvalues
    double sum_e = D.sum();
    // 2) divide each eigenvalue by the sum
    Eigen::VectorXd D_p = D.array() / sum_e;
    D_p.reverseInPlace();
    // 3) add up values until 0.95
    double expl_var_curr = D_p.segment(0, keep_components).sum();
    //  double expl_var_curr = D_p.sum();
    LOG(1, "Using " << keep_components << " components, we can explain " << expl_var_curr << " of the variance.");

    double total = 0.0;
    int num_c;
    for (num_c = 0; num_c < D_p.rows() && total < 0.95; ++num_c)
      {
        total += D_p(num_c);
      }
    LOG(1, "To explain approx. 0.95 of the variance, " << num_c << " components are needed");

    // we want to use as many components to explain 90% of the variance.
    int c = N;
    for (size_t i = 0; i < N; ++i)
      {
        SqrtD(i, i) = 1. / sqrt(D(i) + 0.1);
      }

    // check whether we want to use explained variance criterion
    if (use_explained_variance)
      {
        LOG(1, "Explained variance switch: we use " << num_c << " components instead of " << keep_components);
        c = num_c;
      }
    else
      {
        // check if we want to keep less components
        if (keep_components > 0 && c > keep_components)
          {
            c = keep_components;
          }
      }

    // relevant eigenvector submatrix
    Eigen::MatrixXd E = eigensolver.eigenvectors().block(0, N - c, N, c - drop_high);
    // the final transformation matrix
    Eigen::MatrixXd A;
    // compute tranformation matrix A
    if (back_transform)
      A = E * SqrtD.block(N - c, N - c, c - drop_high, c - drop_high) * E.transpose();
    else
      A = SqrtD.block(N - c, N - c, c - drop_high, c - drop_high) * E.transpose();

    // apply transformation (do this also for test points)
    for (std::vector<Eigen::VectorXd>::iterator k = patches.begin(); k != patches.end(); ++k)
      {
        *k = A * (*k);
      }
    // return the transformation matrix so that it can be used later on
    return A;
  }
  
  Eigen::MatrixXd 
  zca_whitening(std::vector<Eigen::VectorXd> & patches, int num_threads)
  {
    // zca whitening can be simply implemented as a special case of pca whitening
    // where all components are kept and the inputs are back transformed
    // into the original space
    return pca_whitening(patches, -1, 0, false, true, num_threads);
  }


  void
  extract_patches(IplImage *img, std::vector<Eigen::VectorXd> & patches, int patch_size_x, int patch_size_y, int patches_per_img)
  {
    CvSize size = cvGetSize(img);
    assert(size.width > patch_size_x);
    assert(size.height > patch_size_y);
    // set of start positions
    // --> we do not want to sample one position twice
    std::set<int> start_positions;
    int counter = 0;
    int next_x = 0, next_y = 0;
    int patch_total_size = patch_size_x * patch_size_y;
    int num_channels = 0;

    while (counter < patches_per_img)
      {
        num_channels = img->nChannels;
        next_x = drand48() * (size.width - patch_size_x);
        next_y = drand48() * (size.height - patch_size_y);
        int position = next_x + next_y * img->width;
        if (start_positions.find(position) == start_positions.end())
          {
            counter++;
            start_positions.insert(position);
            Eigen::VectorXd patch(patch_size_x * patch_size_y * num_channels);

            if (num_channels == 1)
              {
                extract_at_pos(img, 0, patch, 0, next_y, next_x, patch_size_x, patch_size_y);
              }

            if (num_channels >= 3)
              {
                extract_at_pos(img, 2, patch, 0, next_y, next_x, patch_size_x, patch_size_y);
                extract_at_pos(img, 1, patch, patch_total_size, next_y, next_x, patch_size_x, patch_size_y);
                extract_at_pos(img, 0, patch, 2 * patch_total_size, next_y, next_x, patch_size_x, patch_size_y);
              }

            if (num_channels == 4)
              extract_at_pos(img, 3, patch, 3 * patch_total_size, next_y, next_x, patch_size_x, patch_size_y);

            patches.push_back(patch);
          }
      }
  }

}
