#Libraries
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

#/////////////////////////////////////////////////////////////////////// Data Generation Program ///////////////////////////////////////////////////////////

# Initialize SQLite database
connection = sqlite3.connect("switch_data_same_vendor.db", check_same_thread=False)
cursor = connection.cursor()

# Create tables for KPI records
cursor.execute("""
CREATE TABLE IF NOT EXISTS kpi_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    nwep_id TEXT NOT NULL,
    vendor TEXT NOT NULL,
    model TEXT NOT NULL,
    throughput REAL NOT NULL,
    port_details TEXT NOT NULL,
    metrics TEXT NOT NULL,
    event_type TEXT,
    exception_reason TEXT,
    thread_name TEXT  -- New column for thread name
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    nwep_id TEXT NOT NULL,
    anomaly_detail TEXT NOT NULL,
    severity TEXT NOT NULL
)
""")
connection.commit()

# Function to generate random metrics
def generate_metrics():
    metrics = []
    for label in ["d12", "d21", "d13", "d31"]:
        metrics.append({
            "metric_type": "Delay",
            "label": label,
            "min_value": round(random.uniform(0, 1), 3),
            "max_value": round(random.uniform(0, 1), 3),
            "time_error": round(random.uniform(0, 1), 3),
        })
    return metrics

# Function to generate a KPI record for a vendor
def generate_kpi_record(thread_name, vendor, model):
    throughput_range = {
        "Vendor1": (2000, 8000),
        "Vendor2": (3000, 10000)
    }

    throughput = round(random.uniform(*throughput_range[vendor]), 2)
    ports = [
        {"port_id": 1, "speed": "100G", "type": "Aggregator"},
        {"port_id": 2, "speed": "25G", "type": "Endpoint"},
        {"port_id": 3, "speed": "25G", "type": "Endpoint"},
    ]
    metrics = generate_metrics()

    # Simulate an error for demonstration purposes
    event_type = "Info"
    exception_reason = None
    if random.random() < 0.1:  # 10% chance of error
        event_type = "Error"
        exception_reason = "Random error occurred"
    
    record = {
        "nwep_id": str(uuid.uuid4()),
        "vendor": vendor,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "throughput": throughput,
        "ports": ports,
        "metrics": metrics,
        "event_type": event_type,
        "exception_reason": exception_reason,
        "thread_name": thread_name  # Include thread_name
    }
    return record


# Function to insert a KPI record into the database
def insert_kpi_record(record):
    cursor.execute(
        "INSERT INTO kpi_records (timestamp, nwep_id, vendor, model, throughput, port_details, metrics, event_type, exception_reason, thread_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["timestamp"],
            record["nwep_id"],
            record["vendor"],
            record["model"],
            record["throughput"],
            yaml.dump(record["ports"], default_flow_style=False),
            yaml.dump(record["metrics"], default_flow_style=False),
            record["event_type"],  # Error info
            record["exception_reason"],  # Exception reason
            record["thread_name"],  # Thread name
        )
    )
    connection.commit()


# Function to generate data periodically
def generate_data(thread_name, vendor, model):
    while not stop_event.is_set():
        try:
            kpi_record = generate_kpi_record(thread_name, vendor, model)
            insert_kpi_record(kpi_record)
            print(f"[{thread_name}] Generated KPI record: {kpi_record['nwep_id']}")
        except Exception as e:
            print(f"[{thread_name}] Error generating KPI record: {str(e)}")
        time.sleep(1)  # Simulate data generation delay

if __name__ == "__main__":
    try:
        # Define a shared event to signal threads to stop
        stop_event = threading.Event()

        #Objective 2:
        thread1 = threading.Thread(target=generate_data, args=("Thread1", "Vendor1", "ModelX"))
        thread2 = threading.Thread(target=generate_data, args=("Thread2", "Vendor1", "ModelX"))

        #Objective 3:
        # thread1 = threading.Thread(target=generate_data, args=("Thread1", "Vendor1", "ModelX"))
        # thread2 = threading.Thread(target=generate_data, args=("Thread2", "Vendor2", "ModelY"))

        # Start threads
        thread1.start()
        thread2.start()

        input("Press Enter to stop the program...\n")
        
        # Signal threads to stop
        stop_event.set()

        # Wait for threads to finish
        thread1.join()
        thread2.join()

    except KeyboardInterrupt:
        print("\nProgram interrupted. Stopping threads...")
        stop_event.set()
        thread1.join()
        thread2.join()
    finally:
        try:
            connection.close()
            print("Connection closed.")
        except NameError:
            print("No connection to close.")
        print("Program terminated successfully.")



#///////////////////////////////////////////////////////////// Data Consumption Program ////////////////////////////////////////////////////////

# Initialize SQLite database connection
connection = sqlite3.connect("switch_data_same_vendor.db")
cursor = connection.cursor()

# Function to load KPI data and filter out rows with null values
def load_kpi_data():
    cursor.execute("SELECT timestamp, nwep_id, throughput, metrics, event_type, exception_reason,thread_name FROM kpi_records")
    rows = cursor.fetchall()

    data = []
    for row in rows:
        timestamp, nwep_id, throughput, metrics_yaml, event_type, exception_reason , _ = row
        if (event_type == "Error") or (exception_reason):
            continue  # Skip rows with error info or exception reason

        try:
            metrics = yaml.safe_load(metrics_yaml)
            data.append({
                "timestamp": timestamp,
                "nwep_id": nwep_id,
                "throughput": throughput,
                "delay_d12": metrics[0]["max_value"],
                "delay_d21": metrics[1]["max_value"],
                "delay_d13": metrics[2]["max_value"],
                "delay_d31": metrics[3]["max_value"],
            })
        except yaml.YAMLError as e:
            print(f"Error parsing YAML for record {nwep_id}: {e}")
            continue

    return pd.DataFrame(data)



# Function to log anomalies to the database
def log_anomalies_to_db(anomalies):
    for index, row in anomalies.iterrows():
        cursor.execute(
            """
            INSERT INTO anomalies (timestamp, nwep_id, anomaly_detail, severity)
            VALUES (?, ?, ?, ?)
            """,
            (row.name.isoformat(), row["nwep_id"], f"Anomalous residual: {row['residual']}", "Critical")
        )
    connection.commit()

# Function to analyze time series and detect abrupt changes
def analyze_time_series(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Decompose the delay_d31 time series
    decomposition = seasonal_decompose(df["delay_d31"], model="additive", period=10)
    df["trend"] = decomposition.trend
    df["residual"] = decomposition.resid

    # Detect abrupt changes in the residual component
    threshold = 0.2  # Define a threshold for anomalies
    df["anomaly"] = abs(df["residual"]) > threshold

    anomalies = df[df["anomaly"]]
    print("Detected anomalies:")
    print(anomalies)

    # Log anomalies to the database
    log_anomalies_to_db(anomalies)

    return df, anomalies

# Function to train and evaluate a Random Forest model
def train_random_forest(df):
    # Prepare features and target
    X = df[["throughput", "delay_d12", "delay_d21", "delay_d13"]]
    y = df["delay_d31"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Plot Predicted vs Actual
    plot_predicted_vs_actual_with_lines(y_test.reset_index(drop=True), pd.Series(y_pred))
    
    return model

#Simple multi layer perceptron
def train_mlp(df):
    X = df[["throughput", "delay_d12", "delay_d21", "delay_d13"]]
    y = df["delay_d31"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MLPRegressor): {mse}")

    # Plot Predicted vs Actual
    plot_predicted_vs_actual_with_lines(y_test.reset_index(drop=True), pd.Series(y_pred))

    return model


# Function to visualize Predicted vs Actual values
def plot_predicted_vs_actual_with_lines(y_test, y_pred):
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', marker='o')

    # Plot predicted values
    plt.scatter(range(len(y_pred)), y_pred, color='orange', label='Predicted', marker='o')

    # Connect actual and predicted values with lines
    for i in range(len(y_test)):
        plt.plot([i, i], [y_test.iloc[i], y_pred[i]], color='gray', linestyle='--', linewidth=0.7)

    # Add labels and title
    plt.xlabel('Data Points')
    plt.ylabel('Delay (d31)')
    plt.title('Predicted vs Actual Output')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()





if __name__ == "__main__":
    try:
        # Load data from the database
        df = load_kpi_data()

        if df.empty:
            print("No data available for analysis.")
        else:
            print("Data loaded successfully. Performing analysis...")

            # Perform time series analysis
            df, anomalies = analyze_time_series(df)
            # print('yes')

            # Train and evaluate Random Forest model
            model = train_mlp(df)

            print("Analysis and modeling complete.")


    except Exception as e:
        print(f"Error: {e}")


    finally:
        connection.close()
