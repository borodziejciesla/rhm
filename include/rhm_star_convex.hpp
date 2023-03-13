#ifndef RHM_INCLUDE_RHM_STAR_CONVEX_HPP_
#define RHM_INCLUDE_RHM_STAR_CONVEX_HPP_

#include "rhm.hpp"

namespace eot {
  template <size_t kinematic_state_size, size_t extent_state_size, size_t measurement_size = 2u>
  class RhmStarConvexTracker : RhmTracker<kinematic_state_size, extent_state_size, measurement_size> {
    public:
      explicit RhmStarConvexTracker(const RhmCalibrations<kinematic_state_size> & calibrations)
        : RhmTracker<kinematic_state_size, extent_state_size, measurement_size>(calibrations) {}
      
      virtual ~RhmStarConvexTracker(void) = default;
    
    protected:
    
    private:
  }
}

#endif  //  RHM_INCLUDE_RHM_STAR_CONVEX_HPP_
