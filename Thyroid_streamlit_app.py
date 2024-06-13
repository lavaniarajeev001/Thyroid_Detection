import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px

def get_clean_data():
    data = data = pd.read_csv("Thyroid_Diff.csv")
    data["Gender"] = data["Gender"].map({"M": 1, "F": 2})
    data["Smoking"] = data["Smoking"] .map({"Yes": 1, "No": 2})
    data["Hx Smoking"] = data["Hx Smoking"].map({"Yes": 1, "No": 2})
    data["Hx Radiothreapy"] = data["Hx Radiothreapy"].map({"Yes": 1, "No": 2})
    data["Thyroid Function"] = data["Thyroid Function"].map({
        "Euthyroid": 1,
        "Clinical Hyperthyroidism": 2,
        "Subclinical Hypothyroidism": 3,
        "Clinical Hypothyroidism": 4,
        "Subclinical Hyperthyroidism": 5})
    data["Physical Examination"] = data["Physical Examination"].map({
        "Multinodular goiter": 1,
        "Single nodular goiter-right": 2,
        "Single nodular goiter-left": 3,
        "Normal": 4,
        "Diffuse goiter": 5})
    data["Adenopathy"] = data["Adenopathy"].map({
        "Posterior": 1,
        "Extensive": 3,
        "No": 2,
        "Left": 4,
        "Bilateral": 5,
        "Right": 6})
    data["Pathology"] = data["Pathology"].map({
        "Papillary": 1,
        "Micropapillary": 2,
        "Follicular": 3,
        "Hurthel cell": 4})
    data["Focality"] = data["Focality"].map({"Uni-Focal": 1, "Multi-Focal": 2})
    data["Risk"] = data["Risk"].map({"Low": 1, "Intermediate": 2, "High": 3})
    data["Response"] = data["Response"].map({
        "Excellent": 1,
        "Structural Incomplete": 2,
        "Indeterminate": 3,
        "Biochemical Incomplete": 4})
    data["Recurred"] = data["Recurred"].map({"Yes": 1, "No": 2})
    data["N"] = data["N"].map({"N0": 1, "N1b": 2, "N1a": 3})
    data["M"] = data["M"].map({"M0": 1, "M1": 2})
    data["Stage"] = data["Stage"].map({
        "I": 1,
        "II": 2,
        "IVB": 3,
        "III": 4,
        "IVA": 5})
    data["T"] = data["T"].map({"T2": 1,
                               "T3a": 2,
                               "T1a": 3,
                               "T1b": 4,
                               "T4a": 5,
                               "T3b": 6,
                               "T4b": 7})
    return data

def add_sidebar():
    st.sidebar.header("Thyroid attribute")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Age","Age"),("Gender","Gender"),("Smoking","Smoking"),("Hx Smoking","Hx Smoking"),
        ("Hx Radiothreapy","Hx Radiothreapy"),("Thyroid Function","Thyroid Function"),
        ("Physical Examination","Physical Examination"),("Adenopathy","Adenopathy"),
        ("Pathology","Pathology"),("Focality","Focality"),("T","T"),("N","N"),
        ("M","M"),("Stage","Stage"),("Response","Response"),("Recurred","Recurred")]
    
    input_dict = {}
    
    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=int(0),
            max_value=int(data[key].max())
            )
        
    return input_dict

def get_scaler_values(input_dict,scaler):
    input_array = np.array(list(input_dict.values())).reshape(1,-1)
    scaled_array = scaler.transform(input_array)
    return scaled_array
    
    

def add_prediction(input_data):
    with open("model.pkl", "rb") as model_in:
        model = pickle.load(model_in)
    with open("scaler.pkl", "rb") as scaler_in:
        scaler = pickle.load(scaler_in)
    
    input_scaled = get_scaler_values(input_data, scaler)
    
    prediction = model.predict(input_scaled)
    
    if prediction[0]==1:
        st.write("The risk of having thyroid is low" ) 
    elif prediction[0]==2:
        st.write("The risk of having thyroid is medium")
    elif prediction[0]==3:
        st.write("The risk of having thyroid is high")
    else:
        st.write("A more detailed study is required")
    
def add_graph(input_data): 
    X = ["Age","Gender","Smoking","Hx Smoking","Hx Radiothreapy","Thyroid Function",
         "Physical Examination","Adenopathy","Pathology","Focality","T","N",
         "M","Stage","Response","Recurred"]
    Y = [input_data["Age"],input_data["Gender"],input_data["Smoking"],input_data["Hx Smoking"],
         input_data["Hx Radiothreapy"],input_data["Thyroid Function"],
         input_data["Physical Examination"],input_data["Adenopathy"],
         input_data["Pathology"],input_data["Focality"],input_data["T"],
         input_data["N"],input_data["M"],input_data["Stage"],input_data["Response"],
         input_data["Recurred"]
        ]
    fig = px.bar(x=X, y=Y, labels={'x': 'Attributes', 'y': 'Count'}, 
                 title='Simple Bar Chart')
    return fig
    

def main():
    st.set_page_config(
        page_title="Thyroid dedection app",
        layout="wide",
        initial_sidebar_state="expanded")
    input_data = add_sidebar()
    
    with st.container():
        st.title("Thyroid dedection app")
        st.write("This app is desined for the prediction of having Thyroid based on the criteria shared by the prospect")
     
    col1,col2=st.columns([4,1])
    
    with col1:
        graph_chart = add_graph(input_data)
        st.plotly_chart(graph_chart)
    with col2:
        if st.button("predict"):
            add_prediction(input_data)
        st.subheader("Disclaimer")
        st.write("The predictions are based on the indivisudal results obtained and is not condecend the openion of a medical practitionar")
            
    
if __name__=="__main__":
    main()