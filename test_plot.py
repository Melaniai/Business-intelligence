import matplotlib.pyplot as plt
import pandas as pd

# Skapa testdata
data = {
    'Date': pd.date_range(start='2023-01-01', periods=4, freq='Q'),
    'Net_Revenue_SEK': [1845000000, 1935000000, 2040000000, 1935000000],
    'Anomaly': [False, False, True, False]
}

df = pd.DataFrame(data)

# Plot funktionen
def plot_revenue(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Net_Revenue_SEK'], marker='o', label='Revenue')
    anomalies = df[df['Anomaly']]
    plt.scatter(anomalies['Date'], anomalies['Net_Revenue_SEK'], color='red', label='Anomalies')
    plt.title("Testplot: Boozt Revenue with Anomaly")
    plt.xlabel("Date")
    plt.ylabel("Net Revenue (SEK)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Kör funktionen
plot_revenue(df)

