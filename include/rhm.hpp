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

        scale_.state(0u) = 0.7;
        scale_.covariance(0u, 0u) = 0.08;
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
      StateWithCovariance<1u> scale_;
      double covariance_pseudo_measurement_ = 0.0;
      Eigen::Vector<double, extent_state_size + 3u> crosscovariance_extent_pseudomeasurement_ = Eigen::Vector<double, extent_state_size + 3u>::Zero();
      Eigen::Vector<double, kinematic_state_size> crosscovariance_kinematic_pseudomeasurement_ = Eigen::Vector<double, kinematic_state_size>::Zero();
      Eigen::Matrix<double, kinematic_state_size, measurement_size> crosscovariance_kinematic_measurement_ = Eigen::Matrix<double, kinematic_state_size, measurement_size>::Zero();
      Eigen::Matrix<double, measurement_size, measurement_size> covariance_measurement_ = Eigen::Matrix<double, measurement_size, measurement_size>::Zero();

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
        // Make corrections
        MakeKinematicCorrection(measurement);
        MakeExtentCorrection(measurement);
      }

      void MakeKinematicCorrection(const Measurement & measurement) {
        const auto inversed_measurement_covariance = covariance_measurement_.inverse();

        // Make correction
        state_.kinematic_.state += crosscovariance_kinematic_measurement_ * inversed_measurement_covariance * (measurement.value - h_ * state_.kinematic.state);
        state_.kinematic.covariance -= crosscovariance_kinematic_measurement_ * inversed_measurement_covariance * crosscovariance_kinematic_measurement_.transpose();

        // Force symetricity
        state_.kinematic.covariance = MakeMatrixSymetric<kinematic_state_size>(state_.kinematic.covariance);
      }

      void MakeExtentCorrection(const Measurement & measurement) {
        /* Implements UKF Step */
        // Noise
        Eigen::Vector3d measurement_noise_mean = Eigen::Vector3d::Zero();
        measurement_noise_mean(0u) = scale_.state(0u);

        Eigen::Matrix3d measurement_noise_cov = Eigen::Matrix3d::Zero();
        measurement_noise_cov(0u, 0u) = scale_.covariance(0u, 0u);
        measurement_noise_cov.block<2u, 2u>(1u, 1u) = measurement.covariance;

        // Prepare current extended state
        Eigen::Vector<double, extent_state_size + 3u> p_ukf;
        p_ukf.head(extent_state_size) = state_.extent.state;
        p_ukf.tail<3>() = measurement_noise_mean;

        Eigen::Matrix<double, extent_state_size + 3u, extent_state_size + 3u> c_ukf;
        c_ukf.block<extent_state_size, extent_state_size>(0u, 0u) = state_.extent.covariance;
        c_ukf.block<3u, 3u>(extent_state_size, extent_state_size) = measurement_noise_cov;

        /* Calculate sigma points */
        constexpr auto n = extent_state_size + 3u;
        constexpr auto n_state = extent_state_size;

        const auto lambda = std::pow(alpha_, 2u) * (static_cast<double>(n) + kappa_) - static_cast<double>(n);

        // Calculate weight mean
        constexpr auto sigma_points_number = 2u * n + 1u;
        std::array<double, sigma_points_number> wm;
        wm.at(0u) = lambda / (static_cast<double>(n) + lambda);
        std::fill(wm.begin() + 1u, wm.end(), 1.0 / (2.0 * (static_cast<double>(n) + lambda)));
        // Calculate weight covariance
        std::array<double, sigma_points_number> wc;
        wc.at(0u) = (lambda / (static_cast<double>(n) + lambda)) + (1.0 - std::pow(alpha_, 2u) + beta_);
        std::fill(wc.begin() + 1u, wc.end(), 1.0 / (2.0 * (static_cast<double>(n) + lambda));

        // Calculate Sigm-Points
        const auto sigma_points_cholesky_part = std::sqrt(static_cast<double>(n) + lambda) * c_ukf.llt().matrixL();

        std::array<Eigen::Vector<double, extent_state_size + 3u>, sigma_points_number> sigma_points;
        sigma_points.at(0) = p_ukf;
        for (auto index = 0u; index < n; index++) {
          sigma_points.at(index + 1u) = p_ukf + sigma_points_cholesky_part.col(index);
          sigma_points.at(n + index + 1u) = p_ukf - sigma_points_cholesky_part.col(index);
        }

        // UKF Prediction Step
        std::array<double, sigma_points_number> sigma_points_predicted;
        std::transform(sigma_points.begin(), sigma_points.end(),
          sigma_points_predicted.begin(),
          [](const Eigen::Vector<double, extent_state_size + 3u> & sigma_point){
            return 0.0;
          }
        );

        // Predicted measurement
        std::array<double, sigma_points_number> sigma_points_predicted_weighted;
        std::transform(sigma_points_predicted.begin(), sigma_points_predicted.end(), wm.begin(),
          sigma_points_predicted_weighted.begin(),
          [](const double predicted_sigma_point, const double weight){
            return predicted_sigma_point * weight;
          }
        );
        double z = std::accumulate(sigma_points_predicted_weighted.begin(), sigma_points_predicted_weighted.end(), 0.0);

        // Predicted measurement covariance
        std::array<double, sigma_points_number> sigma_points_predicted_weighted_cov;
        std::transform(sigma_points_predicted.begin(), sigma_points_predicted.end(), wc.begin(),
          sigma_points_predicted_weighted_cov.begin(),
          [=z](const double predicted_sigma_point, const double weight){
            return weight * std::pow(predicted_sigma_point - z, 2u);
          }
        );
        double cov_yy = std::accumulate(sigma_points_predicted_weighted_cov.begin(), sigma_points_predicted_weighted_cov.end(), 0.0);

        // Crosscovariance
        Eigen::Vector<double, extent_state_size + 3u> cov_py = Eigen::Vector<double, extent_state_size + 3u>::Zero();
        for (auto index = 0u; index < sigma_points_number; index++)
          cov_py += wc.at(index) * (sigma_points_predicted.at(index) - z) * (sigma_points.at(index) - state_.extent.state);

        // Kalman Gain
        const auto kalman_gain = cov_py / cov_yy;

        // Estimated state
        state_.extent.state -= kalman_gain * z;
        state_.extent.covariance -= kalman_gain * cov_yy * kalman_gain.transpose();
        state_.extent.covariance = MakeMatrixSymetric<extent_state_size>(state_.extent.covariance)
      }

      double CalculatePseudoMeasurementSquare(const Measurement & measurement, const Eigen::Vector<double, extent_state_size> & shape_parameters, const Eigen::Vector3d & noise) {
        /* Find phi */
        const auto center = h_ * state_.kinematic.state;
        const auto phi = std::atan2(measurement.value(1u) - center(1u), measurement.value(0u) - center(0u)) + 2.0 * std::numbers::pi;
        /* Find Fourier coefficients */
        const auto fourier_coeffs = CalculateFourierCoeffs(phi);
        /* e */
        Eigen::Vector2d e;
        e(0u) = std::cos(phi);
        e(1u) = std::sin(phi);

        const auto s = noise(0u);
        const Eigen::Vector2d v = noise.tail<2>();

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

      void CalculateCrossCorelations(const Measurement & measurement) {
        //
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
