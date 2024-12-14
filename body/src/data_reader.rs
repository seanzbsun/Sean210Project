// library imports
use std::fs::File; // import the File type from the fs module to help work with files
use std::io::{self, BufRead}; // import IO utilities and BufRead trait for buffered reading 

#[derive(Debug)] // implement Debug trait for easy printing
// define a struct to represent a player's data
pub struct PlayerRecord {
    pub attack_wins: usize, // number of attack wins in multiplayer battles
    pub defense_wins: usize, // number of defense wins against multiplayer battle attacks
    pub donations: usize, // number of donations made in a clan
    pub builder_tropies: usize, // number of tropies in the builder hall league
    pub trophies: usize, // number of trophies in multiplayer battles
}

// function to read and parse through dataset from a CSV file
pub fn read_file(path: &str) -> Result<Vec<PlayerRecord>, io::Error> {
    let mut records = Vec::new(); // store parsed player records in a vector

    let file = File::open(path).expect("Could not open file"); // open the file
    let reader = io::BufReader::new(file); // wrap file in buffered reader to read lines efficiently

    for line in reader.lines().skip(1) { // skip the header row
        let line = line?; // unwrap the result of reading a line
        let fields: Vec<&str> = line.split(',').collect(); // split lines by commas into fields

        // parse fields from CSV lineinto PlayerRecord
        let record = PlayerRecord { 
            attack_wins: fields[5].parse().unwrap_or(0), //5
            defense_wins: fields[6].parse().unwrap_or(0), //6
            donations: fields[12].parse().unwrap_or(0), //12
            builder_tropies: fields[15].parse().unwrap_or(0), //15
            trophies: fields[10].parse().unwrap_or(0), //10
        };

        records.push(record); // add parsed record to vector
    }

    Ok(records) // return the vector of parsed records if successful

}

