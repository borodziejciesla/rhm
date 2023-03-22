#ifndef RHM_INCLUDE_RHM_HPP_
#define RHM_INCLUDE_RHM_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "measurement.hpp"
#include "object_state.hpp"
#include "rhm_calibrations.hpp"
#include "helper_functions.hpp"

namespace eot {
  template <size_t kinematic_state_size, size_t extent_state_size, size_t measurement_size = 2u>
  class RhmTracker {
    private:
      static constexpr uint8_t n_ = extent_state_size + 3u;
      static constexpr uint8_t n_state_ = extent_state_size;
      static constexpr uint8_t sigma_points_number_ = 2u * n_ + 1u;
      static constexpr double alpha_ = 1.0;
      static constexpr double beta_ = 0.0;
      static constexpr double kappa_ = 0.0;
      static constexpr double lambda_ = std::pow(alpha_, 2u) * (static_cast<double>(n_) + kappa_) - static_cast<double>(n_);

    public:
      using StateVector = Eigen::Vector<double, kinematic_state_size>;
      using StateMatrix = Eigen::Matrix<double, kinematic_state_size, kinematic_state_size>;
      using Measurement = MeasurementWithCovariance<measurement_size>;
      using MeasurementVector = Eigen::Vector<double, measurement_size>;
      using MeasurementMatrix = Eigen::Matrix<double, measurement_size, measurement_size>;
      using ObjectStateRhm = ObjectState<kinematic_state_size, extent_state_size>;
      using RhmCalibration = RhmCalibrations<kinematic_state_size, extent_state_size>;

    public:
      explicit RhmTracker(const RhmCalibration & calibrations)
        : calibrations_{calibrations} {
        state_ = calibrations_.initial_state;

        kinematic_process_noise_ = ConvertDiagonalToMatrix<kinematic_state_size>(calibrations_.process_noise_kinematic_diagonal);
        extent_process_noise_ = ConvertDiagonalToMatrix<extent_state_size>(calibrations_.process_noise_extent_diagonal);

        scale_.state(0u) = 0.7;
        scale_.covariance(0u, 0u) = 0.08;

        // Observation matrix
        h_(0u, 0u) = 1.0;
        h_(1u, 1u) = 1.0;

        // Prepare UKF
        SetUkfWeights();
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

      const ObjectStateRhm & GetEstimatedState(void) const {
        return state_;
      }

    protected:
      virtual void UpdateKinematic(const double time_delta) = 0;

      void UpdateExtent(void) {};

      void FirstEstimation(const std::vector<Measurement> & measurements) {
        // Find center
        const auto [x_min, x_max] = std::minmax_element(measurements.begin(), measurements.end(),
          [](const Measurement & a, const Measurement & b) {
            return a.value(0u) < b.value(0u);
          }
        );
        const auto [y_min, y_max] = std::minmax_element(measurements.begin(), measurements.end(),
          [](const Measurement & a, const Measurement & b) {
            return a.value(1u) < b.value(1u);
          }
        );

        state_.kinematic.state(0u) = 0.5 * ((*x_min).value(0u) + (*x_max).value(0u));
        state_.kinematic.state(1u) = 0.5 * ((*y_min).value(1u) + (*y_max).value(1u));

        // Estimatet orientation
        auto u_11 = 0.0;
        auto u_20 = 0.0;
        auto u_02 = 0.0;

        for (const auto & measurement : measurements) {
          const auto delta_x = measurement.value(0u) - state_.kinematic.state(0u);
          const auto delta_y = measurement.value(0u) - state_.kinematic.state(1u);

          u_11 += delta_x * delta_y;
          u_20 += std::pow(delta_x, 2);
          u_02 += std::pow(delta_y, 2);
        }
        
        const auto alpha = 0.5 * std::atan2(2.0 * u_11, u_20 - u_02);

        // Estimate size
        using Point = std::pair<double, double>;
        std::vector<Point> points_rotated(measurements.size());
        std::transform(measurements.begin(), measurements.end(), points_rotated.begin(),
          [alpha,this](const Measurement & measurement) {
            const auto delta_x = measurement.value(0u) - state_.kinematic.state(0u);
            const auto delta_y = measurement.value(0u) - state_.kinematic.state(1u);
            const auto c = std::cos(-alpha);
            const auto s = std::sin(-alpha);

            const auto x_rotated = delta_x * c - delta_y * s;
            const auto y_rotated = delta_x * s + delta_y * c;

            return std::make_pair(x_rotated, y_rotated);
          }
        );
        
        const auto [min_x, max_x] = std::minmax_element(points_rotated.begin(), points_rotated.end(),
          [](const Point & a, const Point & b) {
            return a.first < b.first;
          }
        );
        const auto [min_y, max_y] = std::minmax_element(points_rotated.begin(), points_rotated.end(),
          [](const Point & a, const Point & b) {
            return a.second < b.second;
          }
        );

        state_.extent.state(0u) = std::max(0.5 * (max_x->first - min_x->first), 0.5 * (max_y->second - min_y->second));
      }
      
      ObjectStateRhm state_;
      Eigen::Matrix<double, kinematic_state_size, kinematic_state_size> c_kinematic_;

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
        // Calculate pseudomeasurement angle
        CalculatePhiPseudomeasurementAngle(measurement);

        // Make corrections
        MakeKinematicCorrection(measurement);
        MakeExtentCorrection(measurement);
      }

      void MakeKinematicCorrection(const Measurement & measurement) {
        // Calculate covariances
        CalculateMeasurementCovariance(measurement);
        CalculateStateMeasurementCroscovariance(measurement);

        // Inverse measurement covariance
        const auto inversed_measurement_covariance = covariance_measurement_.inverse();

        // Make correction
        state_.kinematic.state += crosscovariance_kinematic_measurement_ * inversed_measurement_covariance * (measurement.value - h_ * state_.kinematic.state);
        state_.kinematic.covariance -= crosscovariance_kinematic_measurement_ * inversed_measurement_covariance * crosscovariance_kinematic_measurement_.transpose();

        // Force symetricity
        state_.kinematic.covariance = MakeMatrixSymetric<kinematic_state_size>(state_.kinematic.covariance);
      }

      void MakeExtentCorrection(const Measurement & measurement) {
        /* Noise */
        static Eigen::Vector3d measurement_noise_mean = Eigen::Vector3d::Zero();
        measurement_noise_mean(0u) = scale_.state(0u);

        static Eigen::Matrix3d measurement_noise_cov = Eigen::Matrix3d::Zero();
        measurement_noise_cov(0u, 0u) = scale_.covariance(0u, 0u);
        measurement_noise_cov.block<2u, 2u>(1u, 1u) = measurement.covariance;

        /* Prepare current extended state */
        p_ukf_.head(extent_state_size) = state_.extent.state;
        p_ukf_.tail(3u) = measurement_noise_mean;

        //c_ukf_ = state_.extent.covariance;
        c_ukf_.block(0u, 0u, extent_state_size, extent_state_size) = state_.extent.covariance;
        c_ukf_.block(extent_state_size, extent_state_size, 3u, 3u) = measurement_noise_cov;

        /* Calculate sigma points */
        CalculateSigmaPoint();
        PredictSigmaPointsPseudomeasurements(measurement);
        const auto z = CalculatePredictedPseudomeasurement();
        const auto cov_zz = CalculatePseudomeasurementCovariance(z);
        const auto cov_pz = CalculateShapePseudomeasurementCroscovariance(z);       

        /* Kalman Gain */
        const auto kalman_gain = cov_pz / cov_zz;

        /* Estimated state */
        state_.extent.state -= kalman_gain * z;
        state_.extent.covariance -= kalman_gain * cov_zz * kalman_gain.transpose();
        state_.extent.covariance = MakeMatrixSymetric<extent_state_size>(state_.extent.covariance);
      }

      void CalculatePhiPseudomeasurementAngle(const Measurement & measurement) {
        const auto center = h_ * state_.kinematic.state;
        phi_ = std::atan2(measurement.value(1u) - center(1u), measurement.value(0u) - center(0u));// + 2.0 * std::numbers::pi;?
      }

      double EvaluateRadialFunction(const Eigen::Vector<double, extent_state_size> & fourier_coefficient, const Eigen::Vector<double, extent_state_size> & shape_parameters) const {
        const auto radius_matrix = fourier_coefficient.transpose() * shape_parameters;
        return radius_matrix(0u);
      }

      double CalculatePseudoMeasurementSquare(const Measurement & measurement, const Eigen::Vector<double, extent_state_size + 3u> & shape_parameters) {
        /* Find phi */
        const auto center = h_ * state_.kinematic.state;
        const auto phi = std::atan2(measurement.value(1u) - center(1u), measurement.value(0u) - center(0u));// + 2.0 * std::numbers::pi;

        /* Find Fourier coefficients */
        const auto fourier_coeffs = CalculateFourierCoeffs(phi);
        
        /* e */
        Eigen::Vector2d e;
        e(0u) = std::cos(phi);
        e(1u) = std::sin(phi);

        const auto s = shape_parameters(extent_state_size);
        const Eigen::Vector2d v = shape_parameters.tail(2u);

        Eigen::Vector<double, extent_state_size> shape = shape_parameters.head(extent_state_size);

        const auto radius = EvaluateRadialFunction(fourier_coeffs, shape);

        /* Pseudomeasurement */
        const auto pseudomeasurement = std::pow((center - measurement.value).norm(), 2u)
          - (std::pow(s, 2u) * std::pow(radius, 2u)
          + 2.0 * s * radius * e.transpose() * v
          + std::pow(v.norm(), 2u));

        return pseudomeasurement;
      }

      Eigen::Vector<double, extent_state_size> CalculateFourierCoeffs(const double phi) {
        static Eigen::Vector<double, extent_state_size> fourier_coeffs;

        fourier_coeffs(0u) = 0.5;
        for (auto index = 1u; index <= (extent_state_size - 1u) / 2u; index++) {
          fourier_coeffs(1u + 2u * (index - 1u)) = std::cos(static_cast<double>(index) * phi);
          fourier_coeffs(1u + 2u * (index - 1u) + 1u) = std::sin(static_cast<double>(index) * phi);
        }

        return fourier_coeffs;
      }

      void CalculateMeasurementCovariance(const Measurement & measurement) {
        const auto u = CalculateExtentPartOfMeasurementModel(phi_);

        covariance_measurement_ = h_ * state_.kinematic.covariance * h_.transpose()
          + u * (1.0 / 12.0) * u.transpose()
          + measurement.covariance;
      }

      void CalculateStateMeasurementCroscovariance(const Measurement & measurement) {
        crosscovariance_kinematic_measurement_ = state_.kinematic.covariance * h_.transpose();
      }

      double CalculatePseudomeasurementCovariance(const double z) {
        std::array<double, sigma_points_number_> sigma_points_predicted_weighted_cov;
        std::transform(sigma_points_predicted_.begin(), sigma_points_predicted_.end(), wc_.begin(),
          sigma_points_predicted_weighted_cov.begin(),
          [z](const double predicted_sigma_point, const double weight){
            return weight * std::pow(predicted_sigma_point - z, 2u);
          }
        );
        return std::accumulate(sigma_points_predicted_weighted_cov.begin(), sigma_points_predicted_weighted_cov.end(), 0.0);
      }

      Eigen::Vector<double, extent_state_size> CalculateShapePseudomeasurementCroscovariance(const double z) {
        Eigen::Vector<double, extent_state_size> cov_py = Eigen::Vector<double, extent_state_size>::Zero();
        for (auto index = 0u; index < sigma_points_number_; index++)
          cov_py += wc_.at(index) * (sigma_points_predicted_.at(index) - z) * (sigma_points_.at(index).head(extent_state_size) - state_.extent.state);
        return cov_py;
      }

      void SetUkfWeights(void) {
        // Calculate weight mean
        wm_.at(0u) = lambda_ / (static_cast<double>(n_) + lambda_);
        std::fill(wm_.begin() + 1u, wm_.end(), 1.0 / (2.0 * (static_cast<double>(n_) + lambda_)));
        
        // Calculate weight covariance
        wc_.at(0u) = (lambda_ / (static_cast<double>(n_) + lambda_)) + (1.0 - std::pow(alpha_, 2u) + beta_);
        std::fill(wc_.begin() + 1u, wc_.end(), 1.0 / (2.0 * (static_cast<double>(n_) + lambda_)));
      }

      void CalculateSigmaPoint(void) {
        const Eigen::Matrix<double, extent_state_size + 3u, extent_state_size + 3u> chol = c_ukf_.llt().matrixL();
        const auto sigma_points_cholesky_part = std::sqrt(static_cast<double>(n_) + lambda_) * chol;

        sigma_points_.at(0) = p_ukf_;
        for (auto index = 0u; index < n_; index++) {
          sigma_points_.at(index + 1u) = p_ukf_ + sigma_points_cholesky_part.col(index);
          sigma_points_.at(n_ + index + 1u) = p_ukf_ - sigma_points_cholesky_part.col(index);
        }
      }

      void PredictSigmaPointsPseudomeasurements(const Measurement & measurement) {
        std::transform(sigma_points_.begin(), sigma_points_.end(),
          sigma_points_predicted_.begin(),
          [measurement, this](const Eigen::Vector<double, extent_state_size + 3u> & sigma_point){
            return CalculatePseudoMeasurementSquare(measurement, sigma_point);
          }
        );
      }

      double CalculatePredictedPseudomeasurement(void) {
        std::array<double, sigma_points_number_> sigma_points_predicted_weighted;
        std::transform(sigma_points_predicted_.begin(), sigma_points_predicted_.end(), wm_.begin(),
          sigma_points_predicted_weighted.begin(),
          [](const double predicted_sigma_point, const double weight){
            return predicted_sigma_point * weight;
          }
        );
        return std::accumulate(sigma_points_predicted_weighted.begin(), sigma_points_predicted_weighted.end(), 0.0);
      }

      Eigen::Vector2d CalculateExtentPartOfMeasurementModel(const double phi) {
        static Eigen::Vector2d u = Eigen::Vector2d::Zero();

        const auto fourier_coeffs = CalculateFourierCoeffs(phi);
        const auto radius = EvaluateRadialFunction(fourier_coeffs, state_.extent.state);

        u(0u) = radius * std::cos(phi);
        u(1u) = radius * std::sin(phi);

        return u;
      }

      RhmCalibration calibrations_;

      bool is_initialized_ = false;
      double prev_timestamp_ = 0.0;

      Eigen::Matrix<double, 2u, kinematic_state_size> h_ = Eigen::Matrix<double, 2u, kinematic_state_size>::Zero();

      Eigen::Matrix<double, kinematic_state_size, kinematic_state_size> kinematic_process_noise_ = Eigen::Matrix<double, kinematic_state_size, kinematic_state_size>::Zero();
      Eigen::Matrix<double, extent_state_size, extent_state_size> extent_process_noise_ = Eigen::Matrix<double, extent_state_size, extent_state_size>::Zero();

      double phi_ = 0.0;

      Eigen::Matrix<double, extent_state_size + 3u, extent_state_size + 3u> c_ukf_;
      Eigen::Vector<double, extent_state_size + 3u> p_ukf_;
      std::array<double, sigma_points_number_> wm_;
      std::array<double, sigma_points_number_> wc_;
      std::array<Eigen::Vector<double, extent_state_size + 3u>, sigma_points_number_> sigma_points_;
      std::array<double, sigma_points_number_> sigma_points_predicted_;
  };
} //  namespace eot

#endif  //  RHM_INCLUDE_RHM_HPP_
