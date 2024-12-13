// using my two other modules for reading in the csv and computing MSE and MAE
mod data_reader;
mod metrics;

// libraries
use linfa::Dataset; // data rep for ML
use linfa::traits::Fit; // fitting model trait
use linfa::prelude::*; // general linfa prelude
use linfa_linear::LinearRegression; // implementing linear regression model
use ndarray::Array2; // for manipulating feature matrices

fn main() {
    // file path var
    let file_path = "/Users/seansun/Documents/ds210/project/Sean210Project/body/src/CoC.csv";

    // read data from CSV using data_reader module
    match data_reader::read_file(file_path) {
        Ok(data) => {
            println!("Successfully read data!");

            // Prepare feature matrix (x) and target vector (y)
            let mut x_data = Vec::new();
            let mut y_data = Vec::new();
            
            // loop through each feature in the record and push to feature matrix and target vector
            for record in data {
                x_data.push(vec![
                    record.attack_wins as f64,
                    record.defense_wins as f64,
                    record.donations as f64,
                    record.builder_tropies as f64,
                ]);
                y_data.push(record.trophies as f64);
            }
            
            // convert feature and target matrix/vector 
            let x = Array2::from_shape_vec((x_data.len(), x_data[0].len()), x_data.concat()).unwrap();
            let y = ndarray::Array1::from_vec(y_data);

            let correlations = metrics::calculate_correlations(&x, &y);
            for (i, correlation) in correlations.iter().enumerate() {
                println!("Correlation between feature {} and trophies: {:.3}", i + 1, correlation);
            }

            // create a dataset that's cloned off of the original data
            let dataset = Dataset::new(x.clone(), y.clone());

            // fit linear regression model 
            let lin_reg = LinearRegression::new();
            match lin_reg.fit(&dataset) {
                Ok(model) => {
                    println!("Model trained successfully!");

                    // print coefficients and intercept
                    println!("Coefficients: {:?}", model.params());
                    println!("Intercept: {:?}", model.intercept());

                    // predict using the fitted model
                    let predictions = model.predict(&x);
                    println!("Predictions: {:?}", predictions);
                    
                    // calculate MAE and MSE
                    let mae = metrics::calculate_mae(&y, &predictions);
                    let mse = metrics::calculate_mse(&y, &predictions);
            
            // print metrics
            println!("Mean Absolute Error (MAE): {:.2}", mae);
            println!("Mean Squared Error (MSE): {:.2}", mse);
                }
                Err(e) => eprintln!("Failed to train the model: {}", e),
            }
        }
        // if file reading fails, print an error message
        Err(e) => eprintln!("Error reading data: {}", e),
    }
}

// #[cfg(test)]

//     #[derive(Debug, PartialEq, Eq)]

//     #[test]