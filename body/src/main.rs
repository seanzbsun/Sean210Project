// using my two other modules for reading in the csv and computation
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
                    
                    // calculate correlation
                    let correlations = metrics::calculate_correlations(&x, &y);
                    for (i, correlation) in correlations.iter().enumerate() {
                        println!("Correlation between feature {} and trophies: {:.3}", i + 1, correlation);
                    }
            
            // print metrics
            println!("Mean Absolute Error (MAE): {:.2}", mae);
            println!("Mean Squared Error (MSE): {:.2}", mse);
                }
                Err(e) => eprintln!("Failed to train the model: {}", e)
            }
        }
        // if file reading fails, print an error message
        Err(e) => eprintln!("Error reading data: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_metrics_calculations() {
        // created a sample dataset
        let features = array![[5.0, 3.0, 10.0, 2.0], [8.0, 6.0, 15.0, 3.0]];
        let trophies = array![200.0, 450.0];

        // test correlation calculation
        let correlations = metrics::calculate_correlations(&features, &trophies);
        assert_eq!(correlations.len(), 4);

        // test MAE calculation
        let predictions = array![200.0, 450.0];
        let mae = metrics::calculate_mae(&trophies, &predictions);
        let mse = metrics::calculate_mse(&trophies, &predictions);

        // assertions checking if the calculated metrics are near zero
        assert!((mae - 0.0).abs() < 1e-6); // both should be near zero
        assert!((mse - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_file_success() {
        // create temporary file that mimics file and stores given 3 lines of data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "header1,header2,header3,header4,header5,attack_wins,defense_wins,dummy1,dummy2,dummy3,trophies,dummy4,donations,dummy5,dummy6,builder_tropies")
            .unwrap();
        writeln!(temp_file, "0,0,0,0,0,5,3,0,0,0,1000,0,50,0,0,10").unwrap();
        writeln!(temp_file, "0,0,0,0,0,8,6,0,0,0,1200,0,60,0,0,15").unwrap();

        let result = data_reader::read_file(temp_file.path().to_str().unwrap());
        assert!(result.is_ok()); // check if the file reading is successful, should succeed
        let records = result.unwrap();
        assert_eq!(records.len(), 2); // check if the correct number of records is read
        assert_eq!(records[0].trophies, 1000); // check if the first record's trophies match
    }

    #[test]
    // test file reading failure
    #[should_panic] // test panic during file reading, aside from invalid path
    fn test_read_file_invalid_format() {
        // temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "header1,header2,header3,header4,header5,attack_wins,defense_wins,dummy1,dummy2,dummy3,trophies,dummy4,donations,dummy5,dummy6,builder_tropies")
            .unwrap();
        writeln!(temp_file, "invalid,data").unwrap();

        // attempt to read the file
        let result = data_reader::read_file(temp_file.path().to_str().unwrap());
        println!("{:?}",result);
        // check if file reading errors
        assert!(result.is_err());
    }
}
