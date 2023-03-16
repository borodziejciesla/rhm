#ifndef RHM_TESTS_MODEL_CV_HPP_
#define RHM_TESTS_MODEL_CV_HPP_

#include "rhm.hpp"

namespace eot {
  /* x, y, vx, vy */
  class ModelCv : public RhmTracker<4u, 5u> {
    public:
      explicit ModelCv(const RhmCalibrations<4u, 5u> & calibrations) 
        : RhmTracker<4u, 5u>(calibrations) {
        // TODO
      }

    protected:
      void UpdateKinematic(const double time_delta) {
        // Update helpers
        SetTransitionMatrix(time_delta);

        // Update kinematic
        state_.kinematic.state = transition_matrix_ * state_.kinematic.state;
        state_.kinematic.covariance = transition_matrix_ * state_.kinematic.covariance * transition_matrix_.transpose() + time_delta * c_kinematic_;
      }
    
    private:
      void SetTransitionMatrix(const double time_delta) {
        //x
        transition_matrix_(0u, 0u) = 1.0;
        transition_matrix_(0u, 2u) = time_delta;
        // y
        transition_matrix_(1u, 2u) = 1.0;
        transition_matrix_(1u, 3u) = time_delta;
        // vx
        transition_matrix_(2u, 1u) = 1.0;
        // vy
        transition_matrix_(3u, 3u) = 1.0;
      }

      StateMatrix transition_matrix_ = StateMatrix::Zero();
  };
} //  namespace eot

#endif  //  RHM_TESTS_MODEL_CV_HPP_
