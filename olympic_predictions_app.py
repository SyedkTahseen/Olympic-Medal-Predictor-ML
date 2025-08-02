import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Page configuration
st.set_page_config(
    page_title="Olympic Medal Predictor",
    page_icon="🏅",
    layout="wide"
)

# Load and cache data
@st.cache_data
def load_data():
    teams = pd.read_csv('teams.csv')
    return teams

@st.cache_resource
def train_model():
    teams = load_data()
    teams_ml = teams.dropna().copy()
    
    features = ['athletes', 'age', 'height', 'weight', 'prev_medals', 'prev_3_medals']
    X = teams_ml[features]
    y = teams_ml['medals']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features, teams_ml

# Main app
def main():
    st.title("🏅 Olympic Medal Predictor")
    st.markdown("**Predict Olympic medal counts using historical data and team characteristics**")
    
    # Load model and data
    model, features, teams_ml = train_model()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🔮 Make Predictions", "📊 Model Performance", "🏆 Historical Analysis", "📈 Country Comparison"])
    
    if page == "🔮 Make Predictions":
        prediction_page(model, features, teams_ml)
    elif page == "📊 Model Performance":
        performance_page(model, features, teams_ml)
    elif page == "🏆 Historical Analysis":
        historical_page(teams_ml)
    elif page == "📈 Country Comparison":
        comparison_page(teams_ml)

def prediction_page(model, features, teams_ml):
    st.header("🔮 Make Medal Predictions")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Characteristics")
        athletes = st.slider("Number of Athletes", 1, 1000, 50)
        age = st.slider("Average Age", 16.0, 40.0, 25.0, 0.1)
        height = st.slider("Average Height (cm)", 150.0, 200.0, 175.0, 0.1)
        weight = st.slider("Average Weight (kg)", 40.0, 120.0, 70.0, 0.1)
    
    with col2:
        st.subheader("Historical Performance")
        prev_medals = st.slider("Previous Olympics Medals", 0, 200, 10)
        prev_3_medals = st.slider("3-Olympics Average Medals", 0.0, 150.0, 8.0, 0.1)
    
    # Make prediction
    if st.button("🎯 Predict Medals", type="primary"):
        input_data = np.array([[athletes, age, height, weight, prev_medals, prev_3_medals]])
        prediction = model.predict(input_data)[0]
        
        st.success(f"🏅 **Predicted Medal Count: {prediction:.1f} medals**")
        
        # Create prediction visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Medals"},
            delta = {'reference': prev_medals},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "gold"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 30], 'color': "silver"},
                        {'range': [30, 100], 'color': "gold"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 50}}))
        
        st.plotly_chart(fig, use_container_width=True)

def performance_page(model, features, teams_ml):
    st.header("📊 Model Performance Analysis")
    
    # Split data for evaluation
    X = teams_ml[features]
    y = teams_ml['medals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.3f}", "Excellent" if r2 > 0.8 else "Good")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}", "medals")
    with col3:
        st.metric("Predictions within 5 medals", f"{((np.abs(y_test - y_pred) <= 5).mean()*100):.1f}%")
    
    # Actual vs Predicted scatter plot
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={'x': 'Actual Medals', 'y': 'Predicted Medals'},
        title="Actual vs Predicted Medal Counts"
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=max(y_test), y1=max(y_test), 
                  line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance in Medal Prediction")
    st.plotly_chart(fig, use_container_width=True)

def historical_page(teams_ml):
    st.header("🏆 Historical Olympic Analysis")
    
    # Top countries by total medals
    country_medals = teams_ml.groupby('team')['medals'].sum().sort_values(ascending=False).head(15)
    
    fig = px.bar(x=country_medals.values, y=country_medals.index, orientation='h',
                 title="Top 15 Countries by Total Olympic Medals",
                 labels={'x': 'Total Medals', 'y': 'Country'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Medal trends over time
    yearly_medals = teams_ml.groupby('year')['medals'].sum()
    fig = px.line(x=yearly_medals.index, y=yearly_medals.values,
                  title="Total Olympic Medals Over Time",
                  labels={'x': 'Year', 'y': 'Total Medals'})
    st.plotly_chart(fig, use_container_width=True)

def comparison_page(teams_ml):
    st.header("📈 Country Comparison Tool")
    
    # Country selector
    countries = st.multiselect("Select countries to compare:", 
                              sorted(teams_ml['team'].unique()),
                              default=['United States', 'China', 'Russia'])
    
    if countries:
        filtered_data = teams_ml[teams_ml['team'].isin(countries)]
        
        # Medal progression over time
        fig = px.line(filtered_data, x='year', y='medals', color='team',
                      title="Medal Count Progression",
                      labels={'year': 'Year', 'medals': 'Medals'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison table
        comparison_data = filtered_data.groupby('team').agg({
            'medals': ['sum', 'mean', 'max'],
            'athletes': 'mean',
            'year': ['min', 'max']
        }).round(2)
        
        st.subheader("Country Statistics")
        st.dataframe(comparison_data, use_container_width=True)

if __name__ == "__main__":
    main()
