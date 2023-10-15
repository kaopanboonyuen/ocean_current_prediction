# GISTDA Ocean Current Prediction (AI Model)

Deep Autoregressive Networks (LSTM and Transformer) for Ocean Current Model

![](logo_gist.png)

## Description

This Python script allows you to predict ocean currents based on different models. You can specify the model, date, hour, latitude, and longitude as command line arguments.

## Usage

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**:

   To predict ocean currents, use the following command line arguments:

   - `--model`: Choose the ocean current classifier. Options: "None", "GI21-Model", "GI31-Model", "GI41-Model", "GULF3-Model", "GULF4-Model", "GULF-Model".
   - `--date`: Specify the date for ocean current prediction in YYYY-MM-DD format.
   - `--hour`: Specify the hour to start prediction (0-23).
   - `--latitude`: Specify the latitude for prediction.
   - `--longitude`: Specify the longitude for prediction.

   **Examples**:

   - Example 1: Predict ocean currents using "GULF3-Model" for latitude 12.1146134 and longitude 100.8672236 on December 7, 2022, starting at hour 5.

     ```bash
     python ocean_current_prediction.py --model "GULF3-Model" --date "2022-12-7" --hour 5 --latitude 12.1146134 --longitude 100.8672236
     ```

   - Example 2: Predict ocean currents using "GI41-Model" for latitude 10.0 and longitude 100.0 on September 20, 2023, starting at hour 14.

     ```bash
     python ocean_current_prediction.py --model "GI41-Model" --date "2023-09-20" --hour 14 --latitude 10.0 --longitude 100.0
     ```

   - Example 3: Predict ocean currents without specifying a model (None) for latitude 8.0 and longitude 102.0 on January 1, 2023, starting at hour 9.

     ```bash
     python ocean_current_prediction.py --model "None" --date "2023-01-01" --hour 9 --latitude 8.0 --longitude 102.0
     ```

   - Example 4: Predict ocean currents using "GULF-Model" for latitude 9.5 and longitude 101.5 on November 15, 2022, starting at hour 16.

     ```bash
     python ocean_current_prediction.py --model "GULF-Model" --date "2022-11-15" --hour 16 --latitude 9.5 --longitude 101.5
     ```

   - Example 5: Predict ocean currents using "GI31-Model" for latitude 11.0 and longitude 100.5 on April 5, 2023, starting at hour 12.

     ```bash
     python ocean_current_prediction.py --model "GI31-Model" --date "2023-04-05" --hour 12 --latitude 11.0 --longitude 100.5
     ```

3. **View Results**:

   The script will provide predictions for ocean currents based on your input parameters. The results will be displayed in the console.

---

You can save this README in your project's root directory to provide clear instructions on how to run the code with various examples.