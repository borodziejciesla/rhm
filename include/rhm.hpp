#ifndef RHM_INCLUDE_RHM_HPP_
#define RHM_INCLUDE_RHM_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <Eigen/Dense>

#include "measurement.hpp"
#include "rhm_calibrations.hpp"
#include "object_state.hpp"
#include "../components/helpers/helper_functions.hpp"

namespace eot {
  template <size_t kinematic_state_size, size_t extent_state_size, size_t measurement_size = 2u>
  class RhmTracker {
    public:
      using StateVector = Eigen::Vector<double, kinematic_state_size>;
      using StateMatrix = Eigen::Matrix<double, kinematic_state_size, kinematic_state_size>;
      using Measurement = MeasurementWithCovariance<measurement_size>;
      using MeasurementVector = Eigen::Vector<double, measurement_size>;
      using MeasurementMatrix = Eigen::Matrix<double, measurement_size, measurement_size>;

    public:
      explicit RhmTracker(const RhmCalibrations<kinematic_state_size> & calibrations)
        : calibrations_{calibrations}
        , c_kinematic_{ConvertDiagonalToMatrix(calibrations_.process_noise_kinematic_diagonal)}
        , c_extent_{ConvertDiagonalToMatrix(calibrations_.process_noise_extent_diagonal)} {
       
        state_ = calibrations_.initial_state;
        c_kinematic_ = ConvertDiagonalToMatrix(calibrations_.process_noise_kinematic_diagonal);
      }
      
      virtual ~RhmTracker(void) = default;

      void Run(const double timestamp, const std::vector<Measurement> & measurements) {
        // Set time delta
        const auto time_delta = timestamp - prev_timestamp_;
        prev_timestamp_ = timestamp;
        // Run algorithm
        // Time update
        if (is_initialized_)
          RunUpdateStep(time_delta);
        else
          FirstEstimation(measurements);
        // Measurement update
        RunCorrectionStep(measurements);
        
        is_initialized_ = true;
      }

      const ObjectState<kinematic_state_size> & GetEstimatedState(void) const {
        return state_;
      }

    protected:
      virtual void UpdateKinematic(const double time_delta) = 0;
      virtual void UpdateExtent(void) = 0;
      virtual void FirstEstimation(const std::vector<Measurement> & measurements) = 0;
      
      ObjectState<kinematic_state_size, extent_state_size> state_;
      Eigen::Matrix<double, kinematic_state_size, kinematic_state_size> c_kinematic_;
      Eigen::Matrix<double, kinematic_state_size, kinematic_state_size> c_extent_;

    private:
      void RunUpdateStep(const double time_delta) {
        /* Update kinematic */
        UpdateKinematic(time_delta);
        /* Update extent state */
        UpdateExtent();
      }

      void RunCorrectionStep(const std::vector<Measurement> & measurements) {
        for (const auto & measurement : measurements)
          MakeOneDetectionCorrection(measurement);
      }

      void MakeOneDetectionCorrection(const Measurement & measurement) {
        SetHelperVariables();
        predicted_measurement_ = h_ * state_.kinematic_state.state;
        innovation_ = measurement.value - predicted_measurement_;

        // Make corrections
        MakeKinematicCorrection(measurement);
        MakeExtentCorrection(measurement);
      }

      void MakeKinematicCorrection(const Measurement & measurement) {
        
      }

      void MakeExtentCorrection(const Measurement & measurement) {
        /* Implements UKF Step */
        Eigen::Vector<double, extent_state_size + 3u> p_ukf;
        Eigen::Matrix<double, extent_state_size + 3u, extent_state_size + 3u> c_ukf;

        /* Calculate sigma points */
        constexpr auto n = extent_state_size + 3u;
        constexpr auto n_state = extent_state_size;

        const auto lambda = std::pow(alpha_, 2u) * (static_cast<double>(n) + kappa_) - static_cast<double>(n);

        // Calculate weight mean
        std::array<double, 2u * n + 1u> wm;
        wm.at(0u) = lambda / (static_cast<double>(n) + lambda);
        std::fill(wm.begin() + 1u, wm.end(), 1.0 / (2.0 * (static_cast<double>(n) + lambda)));
        // Calculate weight covariance
        std::array<double, 2u * n + 1u> wc;
        wc.at(0u) = (lambda / (static_cast<double>(n) + lambda)) + (1.0 - std::pow(alpha_, 2u) + beta_);
        std::fill(wc.begin() + 1u, wc.end(), 1.0 / (2.0 * (static_cast<double>(n) + lambda));

        // CHolesky
      }

      double CalculatePseudoMeasurementSquare(const Measurement & measurement, const Eigen<double, extent_state_size> & shape_parameters) {
        /* Find phi */
        const auto center = h_ * state_.kinematic.state;
        const auto phi = std::atan2(measurement.value(1u) - center(1u), measurement.value(0u) - center(0u)) + 2.0 * std::numbers::pi;
        /* Find Fourier coefficients */
        const auto fourier_coeffs = CalculateFourierCoeffs(phi);
        /* e */
        Eigen::Vector2d e;
        e(0u) = std::cos(phi);
        e(1u) = std::sin(phi);

        const auto s = 0.8;

        Eigen::Vector2d v;
        e(0u) = std::cos(phi);
        e(1u) = std::sin(phi);

        /* Pseudomeasurement */
        const auto pseudomeasurement = std::pow((center - measurement.value).norm(), 2u)
        - (std::pow(s, 2u) * std::pow(fourier_coeffs.transpose() * shape_parameters, 2u) + 2.0 * s * fourier_coeffs.transpose() * shape_parameters * e.transpose() * v + std::pow(v.norm(), 2u));
      }

      Eigen::Vector<double, 2u * extent_state_size + 1u> CalculateFourierCoeffs(const double phi) {
        static Eigen::Vector<double, 2u * extent_state_size + 1u> fourier_coeffs;

        fourier_coeffs(0u) = 0.5;
        for (auto index = 0u; index < extent_state_size - 1u; index++) {
          fourier_coeffs(1u + index * 2u) = std::cos(static_cast<double>(1u + index) * phi);
          fourier_coeffs(1u + index * 2u + 1u) = std::sin(static_cast<double>(1u + index) * phi);
        }

        return fourier_coeffs;
      }

      RhmCalibrations<kinematic_state_size> calibrations_;

      bool is_initialized_ = false;
      double prev_timestamp_ = 0.0;

      Eigen::Vector<double, measurement_size> predicted_measurement_ = Eigen::Vector<double, measurement_size>::Zero();
      Eigen::Vector2d innovation_ = Eigen::Vector2d::Zero();

      Eigen::Matrix<2u, kinematic_state_size> h_ = Eigen::Matrix<2u, kinematic_state_size>::Zero();

      const double alpha_ = 1.0;
      const double beta_ = 0.0;
      const double kappa_ = 0.0;
  };
} //  namespace eot

#endif  //  RHM_INCLUDE_RHM_HPP_
