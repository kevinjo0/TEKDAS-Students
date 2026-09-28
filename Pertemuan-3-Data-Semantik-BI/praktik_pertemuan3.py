import pandas as pd

orders = pd.read_csv("orders.csv")
print(orders.head())
print("Ukuran orders:", orders.shape)
print("Order ID berbeda:", orders["order_id"].nunique())
print("Customer ID berbeda:", orders["customer_id"].nunique())

monthly = pd.read_csv("monthly_revenue.csv")
print("Ukuran monthly_revenue:", monthly.shape)
print(monthly[["year", "month", "orders", "revenue_usd"]].head())

januari_2020 = orders[
    (orders["year"] == 2020) & (orders["month"] == 1)
]
print("Semua order Januari:", len(januari_2020))
print(januari_2020["order_status"].value_counts())
terkirim_januari = januari_2020[
    januari_2020["order_status"] == "Delivered"
]
print("Delivered Januari:", len(terkirim_januari))
print("Revenue Delivered Januari:",
    round(terkirim_januari["total_amount_usd"].sum(), 2))

customers = pd.read_csv("customers.csv")
print("Ukuran customers:", customers.shape)
print("Customer ID berbeda:", customers["customer_id"].nunique())
print(customers[["customer_id", "country", "membership_tier"]].head())

orders_pelanggan = orders.merge(
    customers[["customer_id", "country", "membership_tier"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)
print("Ukuran setelah join:", orders_pelanggan.shape)
print(orders_pelanggan[
    ["order_id", "customer_id", "country", "membership_tier"]
].head())

print("Country kosong:", orders_pelanggan["country"].isna().sum())
print("Tier kosong:", orders_pelanggan["membership_tier"].isna().sum())

order_terkirim = orders_pelanggan[
 orders_pelanggan["order_status"] == "Delivered"
]
pendapatan_negara = (
 order_terkirim.groupby("country")["total_amount_usd"]
 .sum()
 .sort_values(ascending=False)
)
print(pendapatan_negara.head(5))

status_negara = pd.crosstab(
 orders_pelanggan["country"],
 orders_pelanggan["order_status"]
).reindex(
 columns=["Delivered", "Processing", "Cancelled", "Returned"],
 fill_value=0
)
status_negara["Total"] = status_negara.sum(axis=1)
status_negara["Persen Delivered"] = (
 status_negara["Delivered"] / status_negara["Total"] * 100
).round(1)
print(status_negara.sort_values("Total", ascending=False).head(5).to_string())

print("Rating kosong:", orders["customer_rating"].isna().sum())
print("Rating nol:", orders["customer_rating"].eq(0).sum())
print("Rata-rata rating tercatat:", orders["customer_rating"].mean())

produk = pd.read_csv("product_summary.csv")
print("Ukuran product_summary:", produk.shape)
print("Produk berbeda:", produk["product_name"].nunique())
print(produk[
 ["category", "product_name", "total_orders", "total_revenue_usd"]
].head())