// lib imports
use ndarray::{Array1, Array2};

// calculate Mean Absolute Error (MAE)
pub fn calculate_mae(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    actual
        .iter() 
        .zip(predicted.iter()) 
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>() 
        / actual.len() as f64 //
}

// calculate Mean Squared Error (MSE)
pub fn calculate_mse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}

/// calculate correlation coefficients between features and target
pub fn calculate_correlations(features: &Array2<f64>, target: &Array1<f64>) -> Vec<f64> {
    let mut correlations = Vec::new();
    let target_mean = target.mean().unwrap_or(0.0);
    let target_std = target.std(0.0);
    
    for feature_col in features.axis_iter(ndarray::Axis(1)) {
        let feature_mean = feature_col.mean().unwrap_or(0.0);
        let feature_std = feature_col.std(0.0);

        // compute covariance
        let covariance = feature_col.iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - feature_mean) * (y - target_mean))
            .sum::<f64>() / feature_col.len() as f64;

        // compute correlation
        let correlation = if feature_std > 0.0 && target_std > 0.0 {
            covariance / (feature_std * target_std)
        } else {
            0.0
        };
        correlations.push(correlation);
    }
    correlations
}