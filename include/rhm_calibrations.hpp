#ifndef RHM_INCLUDE_MEM_EKF_CALIBRATIONS_HPP_
#define RHM_INCLUDE_MEM_EKF_CALIBRATIONS_HPP_

#include <array>

#include "object_state.hpp"

namespace eot {
template <size_t kinematic_state_size, size_t extent_state_order>
struct RhmCalibrations {
  std::array<double, kinematic_state_size>
      process_noise_kinematic_diagonal;  // Covariance of the process noise for the kinematic state
  std::array<double, 2 * extent_state_order + 1u>
      process_noise_extent_diagonal;  // Covariance of the process noise for the shape parameters

  ObjectState<kinematic_state_size, extent_state_order> initial_state;  // Initial state value
};
}  //  namespace eot

#endif  //  RHM_INCLUDE_MEM_EKF_CALIBRATIONS_HPP_
