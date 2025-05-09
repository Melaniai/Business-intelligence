
from src.data_loader import load_boozt_data
from src.anomaly import detect_anomalies
from src.prediction import train_revenue_model
from src.visualization import plot_revenue
from src.churn_model import predict_churn
from src.forecast import forecast_sales
from src.segmentation import segment_customers
from src.recommender import get_recommendations
from src.content_recommender import recommend_similar_products
from src.sentiment_analysis import analyze_sentiment
import os

# CPU-inställning (för joblib-parallelisering)
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

def main():
    # 🛍️ Produktrekommendationer (Collaborative Filtering)
    print("\n🛍️ Produktrekommendationer för kund 1:")
    recommendations = get_recommendations(customer_id=1)
    print(recommendations)

    # 🧠 Produktrekommendationer (Content-Based Filtering)
    print("\n🧠 Content-based rekommendationer för produkt 101:")
    similar = recommend_similar_products(product_id=101)
    print(similar.to_string(index=False))

    # 📊 Revenue-anomalier
    df = load_boozt_data()
    df = detect_anomalies(df)
    print("\n📊 Intäkter med identifierade anomalier:")
    print(df[['Date', 'Net_Revenue_SEK', 'Anomaly']])
    plot_revenue(df)

    # 📈 Revenue-prediktion
    model, predictions, y_test, error = train_revenue_model(df)
    print(f"\n📈 Genomsnittligt prediktionsfel (MAE): {error:,.0f} SEK ({error/1e6:.2f} MSEK)")

    # 🔁 Churn-prediktion
    model, churn_report, churn_df = predict_churn()
    sorted_df = churn_df.sort_values('PredictedChurn', ascending=False)
    print("\n🔁 Kunder + förutsagd churn (1 = lämnar):")
    print(sorted_df[['Purchases', 'DaysSinceLastPurchase', 'AvgOrderValue', 'PredictedChurn']].to_string(index=False))
    print("\n🔁 Churn Prediction Report:")
    for label, metrics in churn_report.items():
        if isinstance(metrics, dict):
            print(f"Label {label}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}")

    # 🔮 Försäljningsprognos
    forecast, coef, intercept = forecast_sales()
    print(f"\n🔮 Förutsagd försäljning för nästa kvartal: {forecast:,.0f} SEK")
    print(f"Modellkoefficienter: {coef}")

    # 🧩 Kundsegmentering
    segmented_df, kmeans_model = segment_customers()
    print("\n🧩 Kundsegmentering:")
    print(segmented_df[['Purchases', 'DaysSinceLastPurchase', 'AvgOrderValue', 'Segment']].to_string(index=False))

    # 📣 Text- och sentimentanalys
    reviews_df, summary_df = analyze_sentiment()
    print("\n📣 Exempel på recensioner med sentiment:")
    print(reviews_df[['ProductID', 'Review', 'SentimentLabel']].to_string(index=False))
    print("\n📊 Genomsnittligt sentiment per produkt:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
    input("\nTryck [Enter] för att avsluta...")
