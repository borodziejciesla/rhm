#ifndef RHM_INCLUDE_RHM_ELLIPSE_HPP_
#define RHM_INCLUDE_RHM_ELLIPSE_HPP_

#include "rhm.hpp"

namespace eot {
  template <size_t kinematic_state_size, size_t measurement_size = 2u>
  class RhmEllipseTracker : RhmTracker<kinematic_state_size, 3u, measurement_size> {
    public:
      explicit RhmEllipseTracker(const RhmCalibrations<kinematic_state_size> & calibrations)
        : RhmTracker<kinematic_state_size, 3u, measurement_size>(calibrations) {}
      
      virtual ~RhmEllipseTracker(void) = default;
    
    protected:
    
    private:
  }
}

#endif  //  RHM_INCLUDE_RHM_ELLIPSE_HPP_
