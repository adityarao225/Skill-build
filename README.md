# **KPI Analysis Project**

This project is designed to simulate the generation, storage, and analysis of Key Performance Indicator (KPI) data for network switches. The project includes two main components:

1. **Data Generation Program**: Simulates the generation of KPI records and stores them in an SQLite database.
2. **Data Consumption Program**: Loads the stored data, performs anomaly detection, and trains machine learning models for prediction.

---

## **Project Overview**

- **Programming Language**: Python
- **Database**: SQLite
- **Libraries**:
  - `random`, `threading`, `time`, `uuid`, `datetime`
  - `pandas`, `matplotlib`, `yaml`, `sklearn`, `statsmodels`

---

## **Features**

1. **Data Generation**:
   - Simulates random KPI data for multiple vendors and models.
   - Stores generated data in an SQLite database with relevant metadata.
   - Supports multi-threading to simulate concurrent data generation.

2. **Data Storage**:
   - KPI records and detected anomalies are stored in SQLite tables:
     - `kpi_records`: Stores KPI data.
     - `anomalies`: Logs detected anomalies.

3. **Data Analysis**:
   - Loads KPI data from the database.
   - Performs time series analysis to detect abrupt changes or anomalies.
   - Trains and evaluates predictive machine learning models using:
     - Random Forest Regressor
     - Multi-layer Perceptron (MLP) Regressor

4. **Visualization**:
   - Visualizes the predicted vs actual values using scatter plots.

---

## **Project Structure**

- **Data Generation Program**
- **Data Consumption Program**

---

## **Functions**

### **1. Data Generation Program**

#### **`generate_metrics()`**
- Generates random delay metrics (`d12`, `d21`, `d13`, `d31`) for a KPI record.
- Each metric has:
  - `metric_type`: Type of the metric (e.g., Delay).
  - `label`: Metric label.
  - `min_value`, `max_value`, `time_error`: Random values.

#### **`generate_kpi_record(thread_name, vendor, model)`**
- Generates a KPI record for a specific vendor and model.
- Simulates throughput, port details, and error events (10% chance of an error).
- Returns a dictionary containing:
  - `timestamp`, `nwep_id`, `vendor`, `model`, `throughput`
  - `ports`, `metrics`, `event_type`, `exception_reason`, `thread_name`

#### **`insert_kpi_record(record)`**
- Inserts a KPI record into the `kpi_records` SQLite table.
- Converts ports and metrics to YAML format for storage.

#### **`generate_data(thread_name, vendor, model)`**
- Continuously generates KPI records in a loop for a given vendor and model.
- Runs in a separate thread and stops when a `stop_event` is triggered.

---

### **2. Data Consumption Program**

#### **`load_kpi_data()`**
- Loads KPI records from the database while filtering out error events.
- Parses YAML-encoded metrics and returns a DataFrame with the following columns:
  - `timestamp`, `nwep_id`, `throughput`
  - `delay_d12`, `delay_d21`, `delay_d13`, `delay_d31`

#### **`log_anomalies_to_db(anomalies)`**
- Logs detected anomalies into the `anomalies` SQLite table with:
  - `timestamp`, `nwep_id`, `anomaly_detail`, `severity`

#### **`analyze_time_series(df)`**
- Decomposes the `delay_d31` time series into trend and residual components using `seasonal_decompose`.
- Detects anomalies in the residual component based on a predefined threshold.
- Logs detected anomalies to the database and returns the modified DataFrame.

#### **`train_random_forest(df)`**
- Trains a Random Forest Regressor to predict `delay_d31` using:
  - Features: `throughput`, `delay_d12`, `delay_d21`, `delay_d13`
  - Target: `delay_d31`
- Evaluates the model using Mean Squared Error (MSE).
- Visualizes predicted vs actual values.

#### **`train_mlp(df)`**
- Trains a Multi-Layer Perceptron (MLP) Regressor to predict `delay_d31`.
- Uses the same features and target as the Random Forest model.
- Evaluates the model using MSE.
- Visualizes predicted vs actual values.

#### **`plot_predicted_vs_actual_with_lines(y_test, y_pred)`**
- Visualizes predicted vs actual values using scatter plots.
- Connects each actual value to its predicted value with a line for better clarity.

---

## **Usage Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/adityarao225/Skill-build.git
cd Skill-build