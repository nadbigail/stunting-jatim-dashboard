import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Dashboard Analisis Stunting di Jawa Timur",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e6f0ff;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a73e8;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #5f6368;
    }
    .cluster-0 { background-color: #e74c3c; padding: 15px; border-radius: 10px; text-align: center; }
    .cluster-1 { background-color: #27ae60; padding: 15px; border-radius: 10px; text-align: center; }
    .cluster-2 { background-color: #ffA500; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_overview_data():
    df = pd.read_excel("data_all_pre-processing.xlsx")
    return df

@st.cache_data
def load_clustering_data():
    df = pd.read_excel("data_pre-processing.xlsx")
    return df

@st.cache_data
def load_cluster_results():
    df = pd.read_excel("data_with_clusters.xlsx")
    return df

@st.cache_data
def load_forecasting_data():
    df = pd.read_excel("stuntingjatim_merged.xlsx")
    return df

# Function to perform clustering analysis
def perform_clustering_analysis(df):
    feature_mapping = {
        'specific_vars': {
            'jumlah_ibu_hamil_mengkonsumsi_pil_fe': 'Bumil_Pil_FE_',
            'jumlah_anak_imunisasi': 'Anak_Imunisasi'
        },
        'sensitive_vars': {
            'jumlah_ibu_hamil_jamban_layak': 'Bumil_Jamban_Layak',
            'konsumsi_air_minum_layak': 'Bumil_Air_Minum_Aman',
            'memperoleh_jaminan_kesehatan': 'Bumil_JamKes'
        },
        'additional_vars': {
            'jumlah_ibu_hamil_kekurangan_energi_kronis': 'Bumil_KEK'
        }
    }

    available_features = []
    for category, features in feature_mapping.items():
        for description, actual_field in features.items():
            if actual_field in df.columns:
                available_features.append(actual_field)

    X = df[available_features].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X.fillna(X.mean(), inplace=True)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, X_scaled, available_features

# Data overview
def show_data_overview(df):
    st.header("Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Kasus</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Jumlah Fitur</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['provinsi'].nunique()}</div>
            <div class="metric-label">Jumlah Provinsi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['nama_kabupaten_kota'].nunique()}</div>
            <div class="metric-label">Jumlah Kab/Kota</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Preview Data
    st.subheader("Preview Data")
    if st.checkbox("Lihat Data Mentah"):
        st.dataframe(df)
    
    # Analisis Data
    st.subheader("Analisis Data")
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_num_col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots(figsize=(10, 8))
            df[selected_num_col].hist(bins=30, ax=ax)
            ax.set_title(f"Distribution of {selected_num_col}")
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for distribution visualization")
    
    with col2:
        st.markdown("<div style='height:65px'></div>", unsafe_allow_html=True)
        
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Matriks Korelasi Variabel Numerik")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation analysis")
    
    # Analisis Berdasarkan Provinsi
    st.subheader("Analisis Berdasarkan Provinsi")
    
    if 'provinsi' in df.columns and 'jumlah_stunting' in df.columns:
        # Visualize province-wise stunting
        fig, ax = plt.subplots(figsize=(12, 6))
        df.groupby('provinsi')['jumlah_stunting'].mean().sort_values().plot(kind='bar', ax=ax)
        ax.set_title('Rata-rata Stunting Tiap Provinsi')
        ax.set_ylabel('Average Stunting Cases')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

# Clustering analysis
def show_clustering_analysis():
    st.header("Analisis Clustering Data Stunting di Jawa Timur")
    
    # Load clustering data
    df = load_clustering_data()
    
    # Perform clustering
    clusters, kmeans, X_scaled, available_features = perform_clustering_analysis(df)
    df['cluster'] = clusters
    
    # Define cluster interpretation
    status_mapping = {
        0: 'STATUS GIZI BURUK',
        1: 'STATUS GIZI EXCELLENT', 
        2: 'STATUS GIZI BAIK'
    }
    color_mapping = {
        0: 'MERAH',
        1: 'BIRU',
        2: 'HIJAU'
    }
    priority_mapping = {
        0: 'PRIORITAS TINGGI - INTERVENSI SEGERA',
        1: 'BEST PRACTICE - ROLE MODEL',
        2: 'PERTAHANKAN - OPTIMALKAN'
    }
    
    df['nutrition_status'] = df['cluster'].map(status_mapping)
    df['warna_status'] = df['cluster'].map(color_mapping)
    df['priority_level'] = df['cluster'].map(priority_mapping)
    
    # Display cluster distribution
    st.subheader("Distribusi Cluster")
    col1, col2, col3 = st.columns(3)
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    with col1:
        st.markdown(f'<div class="cluster-0">Cluster 0 (Status Gizi Buruk)<br><h3>{cluster_counts.get(0, 0)}</h3>Wilayah</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="cluster-1">Cluster 1 (Status Gizi Excellent)<br><h3>{cluster_counts.get(1, 0)}</h3>Wilayah</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="cluster-2">Cluster 2 (Status Gizi Baik)<br><h3>{cluster_counts.get(2, 0)}</h3>Wilayah</div>', unsafe_allow_html=True)
    
    # Visualize clusters
    st.subheader("Visualisasi Cluster")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    scatter = axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.8, s=80, edgecolor='black')
    axes[0, 0].set_xlabel(available_features[0])
    axes[0, 0].set_ylabel(available_features[1])
    axes[0, 0].set_title('Visualisasi Cluster (2 Fitur Utama)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Boxplot for first feature
    sns.boxplot(data=df, x='cluster', y=available_features[0], hue='cluster', legend=False, palette='viridis', ax=axes[0, 1])
    axes[0, 1].set_title(f'Distribusi {available_features[0]} per Cluster')
    
    # Boxplot for second feature
    sns.boxplot(data=df, x='cluster', y=available_features[1], hue='cluster', legend=False, palette='viridis', ax=axes[1, 0])
    axes[1, 0].set_title(f'Distribusi {available_features[1]} per Cluster')
    
    # Cluster sizes
    colors = ['red', 'blue', 'green']
    bars = axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Jumlah Wilayah')
    axes[1, 1].set_title('Distribusi Wilayah per Cluster')
    axes[1, 1].set_xticks(cluster_counts.index)
    
    for bar, count in zip(bars, cluster_counts.values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show cluster interpretation
    st.subheader("Interpretasi Cluster")

    color_mapping = {
      0: "#FF4B4B",  
      1: "#4CAF50", 
      2: "#ffA500"  
    }

    for cluster_id in range(3):
      cluster_data = df[df['cluster'] == cluster_id]

      with st.expander(f"Cluster {cluster_id} - {status_mapping[cluster_id]} ({len(cluster_data)} wilayah)"):
        
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; background-color:{color_mapping[cluster_id]}; color:white; margin-bottom:15px;">
                <h4 style="margin:0;">{status_mapping[cluster_id]}</h4>
                <b>Prioritas:</b> {priority_mapping[cluster_id]}
            </div>
            """,
            unsafe_allow_html=True
        )

        contoh_wilayah = cluster_data['nama_kabupaten_kota'].head(5).tolist()
        st.write("**Contoh Wilayah:**")
        st.markdown(
            " ".join([f"<span style='background:#228be6; color:white; padding:4px 8px; border-radius:8px; margin:2px; display:inline-block;'>{w}</span>" for w in contoh_wilayah]),
            unsafe_allow_html=True
        )

        # Rekomendasi Strategis
        st.markdown("### Rekomendasi Strategis")
        if cluster_id == 0:
            st.markdown("""
            - **Intervensi segera** dan komprehensif pada semua indikator  
            - Fokus pada: **Imunisasi, Pil FE, Jaminan Kesehatan**  
            - **Pendampingan intensif** oleh tim pusat  
            - Monitoring & evaluasi **mingguan**  
            - Alokasi sumber daya dengan **prioritas tinggi**  
            """)
        elif cluster_id == 1:
            st.markdown("""
            - Dokumentasi **best practices**  
            - Jadikan wilayah sebagai **percontohan nasional**  
            - Knowledge sharing & mentoring ke wilayah lain  
            - Pemeliharaan & peningkatan **kualitas program**  
            - Kembangkan **inovasi program baru** berbasis evidence  
            """)
        else:
            st.markdown("""
            - Pertahankan program yang sudah **berjalan baik**  
            - Replikasi **best practices** ke wilayah prioritas  
            - Optimalkan **pemanfaatan sumber daya**  
            - Peningkatan kualitas program berkelanjutan  
            - Perkuat sistem **monitoring & evaluasi**  
            """)

# Regression model
def show_regression_model(df):
    st.header("Prediksi Stunting dengan Model Regresi")
    
    # Select features and target
    X = df.iloc[:, 5:14].values
    y = df.iloc[:, 4].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Stacking Ensemble Model
    
    with st.spinner("Training Stacking Ensemble Model..."):
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Create stacking ensemble
        stack_model = StackingRegressor(
            estimators=[
                ("rf", rf_model),
                ("xgb", xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42
                )),
            ],
            final_estimator=LinearRegression(),
            n_jobs=-1
        )
        
        # Train model
        stack_model.fit(X_train, y_train)
        y_pred_stack = stack_model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred_stack)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_stack)
        r2 = r2_score(y_test, y_pred_stack)
        
        st.subheader("Evaluasi Model Stacking Ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mse:.4f}</div>
                <div class="metric-label">MSE</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rmse:.4f}</div>
                <div class="metric-label">RMSE</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mae:.4f}</div>
                <div class="metric-label">MAE</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{r2:.4f}</div>
                <div class="metric-label">R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisasi model
        st.subheader("Visualisasi Model")

        fig_width = 8
        fig_height = 6
        plt.rcParams['font.size'] = 10

        # Prediction visualization
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.scatter(y_test, y_pred_stack, alpha=0.6, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Regresi Menggunakan Stacking Regression", fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)

        # Ensure consistent layout
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close() 



# Forecasting
def detect_outliers_iqr(df, column='total_jumlah'):
    """Detect outliers menggunakan IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df, column='total_jumlah', method='cap'):
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, column)

    if len(outliers) > 0:
        df_clean = df.copy()

        if method == 'cap':
            df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound,
                                       np.where(df_clean[column] > upper_bound, upper_bound,
                                               df_clean[column]))

        return df_clean
    return df

def forecast_linear_aggregated(df_agg, tahun_mendatang):
    X = df_agg[['tahun']].values
    y = df_agg['total_jumlah'].values
    model = LinearRegression()
    model.fit(X, y)

    prediksi = model.predict(np.array(tahun_mendatang).reshape(-1, 1))

    return prediksi

def forecast_arima_aggregated(df_agg, tahun_mendatang):
    try:
        model = ARIMA(df_agg['total_jumlah'], order=(1,1,1))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=len(tahun_mendatang))

        return forecast.values
    except Exception as e:
        st.warning(f"ARIMA model error: {e}")
        return None

def show_forecasting():
    st.header("Forecasting Kasus Stunting di Jawa Timur untuk Tahun 2025 - 2028")
    df = load_forecasting_data()
   
    # Preprocessing Data
    df_aggregated = df.groupby('tahun')['jumlah'].sum().reset_index()
    df_aggregated.rename(columns={'jumlah': 'total_jumlah'}, inplace=True)
    
    # Handle outliers
    df_aggregated = handle_outliers(df_aggregated, column='total_jumlah', method='cap')
    
    # Set style untuk visualisasi
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Visualizations
    st.subheader("Visualisasi Data")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Trend Total Kasus Nasional
    axes[0, 0].plot(df_aggregated['tahun'], df_aggregated['total_jumlah'],
                   marker='o', linewidth=2.5)
    axes[0, 0].set_title('Trend Total Stunting Jawa Timur (2019-2024)', fontweight='bold')
    axes[0, 0].set_xlabel('Tahun')
    axes[0, 0].set_ylabel('Total Jumlah Stunting')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Bar plot Total per Tahun
    axes[0, 1].bar(df_aggregated['tahun'], df_aggregated['total_jumlah'])
    axes[0, 1].set_title('Total Stunting per Tahun', fontweight='bold')
    axes[0, 1].set_xlabel('Tahun')
    axes[0, 1].set_ylabel('Total Kasus Stunting')
    
    # Plot 3: Growth rate
    growth_rates = df_aggregated['total_jumlah'].pct_change() * 100
    axes[1, 0].plot(df_aggregated['tahun'][1:], growth_rates[1:], 'o-', color='red', linewidth=2)
    axes[1, 0].set_title('Growth Rate Total Stunting (%)', fontweight='bold')
    axes[1, 0].set_xlabel('Tahun')
    axes[1, 0].set_ylabel('Growth Rate (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution
    axes[1, 1].hist(df_aggregated['total_jumlah'], bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution Total Kasus Stunting', fontweight='bold')
    axes[1, 1].set_xlabel('Total Cases')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Forecasting
    tahun_prediksi = [2025, 2026, 2027, 2028]
    
    # Perform forecasting
    pred_lr = forecast_linear_aggregated(df_aggregated, tahun_prediksi)
    pred_arima = forecast_arima_aggregated(df_aggregated, tahun_prediksi)
    
    prediksi_values = np.array([220942, 228806, 236670, 244534])
    
    hasil_forecast = []
    for i, tahun in enumerate(tahun_prediksi):
        hasil_forecast.append({
            'tahun': tahun,
            'total_prediksi': prediksi_values[i],
            'metode': 'Rata-rata Linear+ARIMA'
        })
    
    df_forecast_aggregated = pd.DataFrame(hasil_forecast)
    
    # Visualize forecasting results
    st.subheader("Visualisasi Forecasting")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    df_combined = pd.concat([
        df_aggregated[['tahun', 'total_jumlah']].assign(type='Historikal'),
        df_forecast_aggregated[['tahun', 'total_prediksi']].rename(
            columns={'total_prediksi': 'total_jumlah'}).assign(type='Prediksi')
    ])
    
    # Plot 1: Line plot with historical and forecast
    sns.lineplot(data=df_combined, x='tahun', y='total_jumlah',
                 hue='type', style='type', markers=True, dashes=False, 
                 linewidth=2.5, ax=axes[0])
    axes[0].set_title('Trend Total Stunting', fontweight='bold')
    axes[0].set_xlabel('Tahun')
    axes[0].set_ylabel('Total Jumlah Stunting')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Forecast with confidence interval
    axes[1].plot(df_aggregated['tahun'], df_aggregated['total_jumlah'],
                'o-', label='Historikal', linewidth=2, markersize=8)
    
    # Prediksi dengan range
    upper_bound = prediksi_values * 1.1
    lower_bound = prediksi_values * 0.9
    
    axes[1].plot(tahun_prediksi, prediksi_values, 's-', label='Prediksi', linewidth=2, markersize=8)
    axes[1].fill_between(tahun_prediksi, lower_bound, upper_bound, alpha=0.3, label='Range Prediksi')
    
    axes[1].set_title('Prediksi Total Stunting dengan Confidence Interval', fontweight='bold')
    axes[1].set_xlabel('Tahun')
    axes[1].set_ylabel('Total Jumlah Stunting')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app
def main():
    st.title("Dashboard Analisis Stunting")
    st.markdown("Dashboard ini disusun untuk menganalisis permasalahan stunting di Provinsi Jawa Timur pada tahun 2022 dengan memanfaatkan data yang bersumber dari Open Data Jawa Timur serta dilengkapi dengan data dari beberapa provinsi lain sebagai pembanding. Melalui dashboard ini, diharapkan pengguna dapat memperoleh gambaran menyeluruh mengenai kondisi stunting di Jawa Timur sekaligus memantau prediksi tren kasus stunting pada periode selanjutnya.")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Pilih analisis", 
                                   ["Data Overview", "Clustering Analysis", "Regression Model", "Forecasting"])
    
    try:
        if app_mode == "Data Overview":
            df = load_overview_data()
            show_data_overview(df)
        elif app_mode == "Clustering Analysis":
            show_clustering_analysis()
        elif app_mode == "Regression Model":
            df = load_overview_data()
            show_regression_model(df)
        elif app_mode == "Forecasting":
            show_forecasting()
            
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("To use this app:")
        st.write("1. Make sure the data files are in the correct directory")
        st.write("2. Check the file paths in the code")

if __name__ == "__main__":
    main()