# importing Standard Libraries
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

# Loading the two datasets and the XGB model
full_set_path = "FullSet.csv"
final_set_lagged_path = "FinalSet_lagged.csv"
model_path = "XGB_lagged.json"

# Loading our data
full_set = pd.read_csv(full_set_path)
final_set_lagged = pd.read_csv(final_set_lagged_path)

# doing the OneHotEncoding here
categorical_columns = ["country"]
encoder = OneHotEncoder(sparse_output=False)
encoded_countries = encoder.fit_transform(full_set[categorical_columns])
encoded_countries_df = pd.DataFrame(encoded_countries, columns=encoder.get_feature_names_out(categorical_columns))

# Renaming and merging the encoded columns
encoded_countries_df.columns = [f"encoded_{col}" for col in encoded_countries_df.columns]
final_set_lagged = pd.concat([final_set_lagged, encoded_countries_df], axis=1)

# Loading our XGBoost model
bst = xgb.Booster()
bst.load_model(model_path)
model_feature_names = bst.feature_names

# Reindex dataset to match model features
final_set_lagged = final_set_lagged.reindex(columns=model_feature_names, fill_value=0)
final_set_lagged = final_set_lagged.apply(pd.to_numeric, errors='coerce')

# Extract war-related columns
war_columns = [col for col in final_set_lagged.columns if col.startswith("war_")]
wars = {col: col.replace("war_", "").replace("_", " ").title() for col in war_columns}

# Streamlit App Interface
st.set_page_config(page_title="Energy Price Projection App", layout="wide")

st.title("🌍 Energy Price Projection App")
st.markdown("Welcome to our app! Use the sidebar to adjust parameters and explore projections.")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Input Settings")
    selected_war = st.selectbox("Select a War", ["None"] + list(wars.values()))
    countries_at_war = st.multiselect("Select Additional Countries at War", full_set["country"].unique())
    selected_country = st.selectbox("Fallback Country", full_set["country"].unique())
    years = st.slider("Projection Years", 1, 5, 1)
    war_duration = st.number_input("War Duration (Years)", 0, 20, 0)
    war_involvement = st.radio("War Involvement Level", ["None", "Territory", "Surroundings", "Involvement"])

# Determine countries to use
selected_countries = []
if selected_war != "None":
    selected_war_column = [key for key, value in wars.items() if value == selected_war][0]
    selected_countries.extend(full_set.loc[full_set[selected_war_column] == 1, "country"].tolist())

selected_countries.extend(countries_at_war)
if not selected_countries:
    selected_countries.append(selected_country)
selected_countries = list(set(selected_countries))

# Display the selected countries which the user chooses
st.info(f"**Selected Countries:** {', '.join(selected_countries)}")

# Prepare input for prediction
def prepare_input_data(countries, years, war_duration, war_involvement):
    input_data = []
    for country in countries:
        if country in full_set["country"].values:
            base_index = full_set[full_set["country"] == country].index[0]
            base_input_row = final_set_lagged.iloc[base_index].copy()

            base_input_row["years"] = years
            base_input_row["war_duration"] = war_duration
            for level in ["None", "Territory", "Surroundings", "Involvement"]:
                base_input_row[f"war_involvement_{level.lower()}"] = 1 if war_involvement == level else 0

            input_data.append(base_input_row)
    return pd.DataFrame(input_data).mean().to_frame().T if len(input_data) > 1 else input_data[0].to_frame().T

input_df = prepare_input_data(selected_countries, years, war_duration, war_involvement)
input_df = input_df.reindex(model_feature_names, axis=1).apply(pd.to_numeric, errors='coerce')
dtest = xgb.DMatrix(data=input_df)
prediction = bst.predict(dtest)


st.header("📊 Energy Price Projections")

# Calculate Projections
years_range = np.arange(1, years + 1)
absolute_changes = prediction + np.random.uniform(-0.005, 0.005, size=len(years_range))
relative_changes = ((absolute_changes - absolute_changes[0]) / absolute_changes[0]) * 100

# Plot Absolute Change
fig1 = px.line(
    x=years_range, 
    y=absolute_changes,
    labels={"x": "Years", "y": "Energy Price (Absolute)"},
    title="Projected Energy Price Over Time",
    markers=True
)
fig1.update_layout(height=500, width=900)  # Set larger graph dimensions
st.plotly_chart(fig1)

# Plot Relative Change
fig2 = px.line(
    x=years_range, 
    y=relative_changes,
    labels={"x": "Years", "y": "Relative Change (%)"},
    title="Projected Energy Price Relative Change",
    markers=True,
    line_shape="spline"
)
fig2.update_layout(height=500, width=900)  # Set larger graph dimensions
st.plotly_chart(fig2)

# Summary Section
st.header("📈 Summary of Predictions")
summary_text = (
    f"The average price of energy is projected to **{'increase' if prediction[0] > 0 else 'decrease'}** "
    f"by approximately **{relative_changes[-1]:.2f}%** over **{years}** years."
)
st.success(summary_text)

# Additional Insights Section
st.markdown("### **Key Takeaways:**")
st.markdown(
    """
    - Countries selected: **{countries}**
    - Projection Years: **{years}**
    - Expected Change: **{change:.2f}%**
    """.format(
        countries=", ".join(selected_countries),
        years=years,
        change=relative_changes[-1],
    )
)

st.markdown("---")
st.markdown("Thank you for using the Energy Price Projection App!")




