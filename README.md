# AQI Prediction with Spiking Neural Network

## AQI Calculation

This notebook calculates the Air Quality Index following the original tutorial on [Kaggle](https://www.kaggle.com/code/rohanrao/calculating-aqi-air-quality-index).

The data is hourly air quality readings from around 230 stations across India, collected between 2015 and 2020. Each row contains 7 pollutant values: PM2.5, PM10, SO2, NOx, NH3, CO, and O3.

How the AQI is computed:
1. Rolling averages are calculated for each pollutant. PM2.5, PM10, SO2, NOx, and NH3 use a 24-hour average. CO and O3 use an 8-hour maximum.
2. Each pollutant is converted to a sub-index on a 0 to 500 scale using predefined breakpoints.
3. The final AQI is the highest sub-index. At least one of PM2.5 or PM10 must be present, and at least 3 out of 7 pollutants must be available.
4. The AQI number is mapped to a bucket: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.

## SNN Changes

After computing the AQI, we try to predict the AQI bucket using SNN.

Three things are different compared to a regular neural network:
- Input: instead of feeding raw numbers, each value is turned into a sequence of 0s and 1s (spikes). Higher pollution means more 1s.
- Neurons: instead of ReLU, we use LIF neurons. Spike once enough input accumulated over time.
- Processing: instead of processing each sample once, the network runs through multiple timesteps in a loop. At the end, it counts how many times each output neuron fired. The class with the most spikes is the prediction.

The rest of the training setup (loss function, optimizer, train/test split) is the same as a standard PyTorch classifier.

### Most recent change

Data changes for the SNN part: we use 21 features instead of 7. The extra features are the rolling averages and sub-indices that were already computed in the AQI calculation. All features are Min-Max normalised to the 0 to 1 range so they can be used as spike firing probabilities. Class weights are applied to the loss function because the dataset is imbalanced (some buckets have far more samples than others).

_This might change in the future, since current prediction is still too low._

## Installation

1. Download the data from the [Kaggle](https://www.kaggle.com/rohanrao/air-quality-data-in-india) and place these two files in a `data/` folder:
   - `station_hour.csv`
   - `stations.csv`

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Open and run `calculating-aqi-with-snn.ipynb`.

## Timesteps and epochs

Training takes a while. A regular neural network processes each batch once. A spiking network processes each batch once per timestep (25 by default), and each step depends on the previous one, so they cannot run in parallel. This means roughly 25 times more work per batch.

Current settings are `NUM_STEPS = 25` and `EPOCHS = 15`.

If you want faster runs, you can lower them. To change them, edit these values in the notebook:
```python
NUM_STEPS = 10
EPOCHS = 5
```
