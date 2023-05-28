#ifndef RHM_INCLUDE_OBJECT_STATE_HPP_
#define RHM_INCLUDE_OBJECT_STATE_HPP_

#include <Eigen/Dense>

namespace eot {
  template <size_t state_size>
  struct StateWithCovariance {
    Eigen::Vector<double, state_size> state = Eigen::Vector<double, state_size>::Zero();
    Eigen::Matrix<double, state_size, state_size> covariance = Eigen::Matrix<double, state_size, state_size>::Zero();
  };

  template <size_t kinematic_state_size, size_t extent_state_size>
  struct ObjectState {
    StateWithCovariance<2u * extent_state_size + 1u> extent;
    StateWithCovariance<kinematic_state_size> kinematic;
  };
} //  namespace eot

#endif  //  RHM_INCLUDE_OBJECT_STATE_HPP_
